import numpy as np
import gzip
import os


def load_mnist(path, kind="train"):
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte.gz")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte.gz")

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 28, 28, 1
        )

    return images, labels


def preprocess_data(images, labels):
    images = images / 255.0  # normalize the pixel values
    labels = np.eye(10)[labels]  # one-hot encode the labels
    return images, labels


def get_dataloaders(batch_size=64):
    X_train, y_train = load_mnist("../mnist/MNIST/raw", kind="train")
    X_test, y_test = load_mnist("../mnist/MNIST/raw", kind="t10k")

    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    def create_batches(X, y, batch_size):
        for i in range(0, len(X), batch_size):
            yield X[i : i + batch_size], y[i : i + batch_size]

    return create_batches(X_train, y_train, batch_size), create_batches(
        X_test, y_test, batch_size
    )
