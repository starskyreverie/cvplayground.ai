import torch
import torch.quantization
from torch import nn
from BabyCNN import BabyCNN
from get_dataloaders import get_dataloaders


def main():
    model = BabyCNN()
    model.eval()
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)

    train_loader, _ = get_dataloaders(batch_size=1)

    # calibrate with the training set
    with torch.no_grad():
        for data, _ in train_loader:
            model(data)

    torch.quantization.convert(model, inplace=True)

    # evaluate the quantized model
    _, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()

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
    print(
        f"Quantized Model - Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()
