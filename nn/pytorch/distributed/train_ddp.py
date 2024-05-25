import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from BabyCNN import BabyCNN
from get_dataloaders import get_dataloaders
from torch import nn, optim


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    model = BabyCNN().cuda(rank)
    model = DDP(model, device_ids=[rank])

    train_loader, test_loader = get_dataloaders()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(rank), target.cuda(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0 and rank == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        if rank == 0:
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.cuda(rank), target.cuda(rank)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)
            accuracy = 100.0 * correct / len(test_loader.dataset)
            print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
