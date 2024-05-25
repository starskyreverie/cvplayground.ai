import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BabyCNN(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.layer_norm1 = nn.LayerNorm([64, 24, 24])  # normalized_shape
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=12 * 12 * 64, out_features=128)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.max_pool1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x


def get_dataloaders():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # (H, W, C) -> (C, H, W)
            transforms.Normalize((0.1307,), (0.3081,)),
        ]  # normalizes the tensor image with mean and standard deviation. These values (0.1307, 0.3081) are precomputed for the MNIST dataset
    )
    train_dataset = datasets.MNIST(
        root="../mnist", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="../mnist", train=False, transform=transform, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def train_mnist():
    train_loader, test_loader = get_dataloaders()
    model = BabyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = (
        nn.CrossEntropyLoss()
    )  # diff between probability distribution and true distribution

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

        # now evaluate the model on the test set
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
    train_mnist()
