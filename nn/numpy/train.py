import numpy as np
from get_dataloaders import get_dataloaders
from BabyCNN import BabyCNN, cross_entropy_loss


def train(model, train_loader, test_loader, num_epochs=10, learning_rate=0.01):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = 0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            print(f"iteration {i}")
            # Reshape X_batch to (batch_size, channels, height, width)
            X_batch = X_batch.transpose(0, 3, 1, 2)
            y_batch_indices = np.argmax(y_batch, axis=1)  # get indices of true labels

            # Forward pass
            output = model.forward(X_batch)
            loss = cross_entropy_loss(output, y_batch_indices)
            train_loss += loss

            # Backward pass
            d_output = output
            d_output[range(X_batch.shape[0]), y_batch_indices] -= 1
            d_output /= X_batch.shape[0]
            model.backward(d_output, learning_rate)

        print(f"Training loss: {train_loss / (i + 1)}")

        # Evaluate on test data
        test_loss, correct = 0, 0
        for X_batch, y_batch in test_loader:
            # Reshape X_batch to (batch_size, channels, height, width)
            X_batch = X_batch.transpose(0, 3, 1, 2)
            y_batch_indices = np.argmax(y_batch, axis=1)  # get indices of true labels
            output = model.forward(X_batch)
            loss = cross_entropy_loss(output, y_batch_indices)
            test_loss += loss
            predictions = np.argmax(output, axis=1)
            correct += np.sum(predictions == y_batch_indices)

        print(f"Test loss: {test_loss / len(test_loader)}")
        print(
            f"Test accuracy: {correct / len(test_loader) / X_batch.shape[0] * 100:.2f}%"
        )


def main():
    np.random.seed(0)
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.01

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = BabyCNN()

    train(
        model,
        train_loader,
        test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    main()
