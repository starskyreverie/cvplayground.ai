import torch
from torch import nn, optim
from torch.nn.utils import prune
from BabyCNN import BabyCNN
from get_dataloaders import get_dataloaders


def main():
    model = BabyCNN()
    train_loader, test_loader = get_dataloaders()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Apply pruning
    parameters_to_prune = (
        (model.conv1, "weight"),
        (model.conv2, "weight"),
        (model.fc1, "weight"),
        (model.fc2, "weight"),
    )
    for module, name in parameters_to_prune:
        prune.l1_unstructured(module, name=name, amount=0.4)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
