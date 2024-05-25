from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
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
