import os
from torchvision import datasets


def download_mnist(data_dir):
    datasets.MNIST(root=data_dir, train=True, download=True)


def main():
    mnist_dir = "./mnist"
    os.makedirs(mnist_dir, exist_ok=True)

    download_mnist(mnist_dir)
    print(f"MNIST dataset downloaded and saved to {mnist_dir}")


if __name__ == "__main__":
    main()
