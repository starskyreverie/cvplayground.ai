import numpy as np
from BabyCNN import BabyCNN
from get_dataloaders import get_dataloaders


def cross_entropy_loss(Y, Y_hat):
    # compute cross-entropy loss
    return -np.mean(np.sum(Y * np.log(Y_hat + 1e-8), axis=1))


def compute_accuracy(Y, Y_hat):
    # compute accuracy
    return np.mean(np.argmax(Y, axis=1) == np.argmax(Y_hat, axis=1))


def backpropagate(params, grads, cache, X, Y):
    # initialize gradients for backpropagation
    m = X.shape[0]  # batch size
    A1, A2, A3, A4 = cache["A1"], cache["A2"], cache["A3"], cache["A4"]

    # gradient for output layer
    dZ4 = A4 - Y  # derivative of loss with respect to Z4
    grads["W4"] = np.dot(A3.T, dZ4) / m  # gradient for W4
    grads["b4"] = np.sum(dZ4, axis=0, keepdims=True).T / m  # gradient for b4

    # gradient for fully connected layer 1
    dA3 = np.dot(dZ4, params["W4"].T)
    dZ3 = dA3 * (A3 > 0)  # derivative of ReLU
    grads["W3"] = np.dot(A2.T, dZ3) / m  # gradient for W3
    grads["b3"] = np.sum(dZ3, axis=0, keepdims=True).T / m  # gradient for b3

    # gradient for conv layer 2
    dA2 = np.dot(dZ3, params["W3"].T)
    dA2 = dA2.reshape(A2.shape)
    dZ2 = dA2 * (A2 > 0)  # derivative of ReLU
    grads["W2"] = np.zeros_like(params["W2"])  # initialize gradient for W2
    grads["b2"] = np.zeros_like(params["b2"])  # initialize gradient for b2
    for i in range(X.shape[0]):  # loop over batch
        grads["W2"] += np.sum(
            dZ2[i, :, :, :, np.newaxis] * A1[i, np.newaxis, :, :, :], axis=(1, 2, 3)
        )
        grads["b2"] += np.sum(dZ2[i, :, :, :], axis=(1, 2))
    grads["W2"] /= X.shape[0]
    grads["b2"] /= X.shape[0]

    # gradient for conv layer 1
    dA1 = np.zeros_like(A1)
    for i in range(X.shape[0]):  # loop over batch
        for h in range(dA1.shape[2]):
            for w in range(dA1.shape[3]):
                dA1[i, :, h, w] = np.sum(
                    dZ2[i, :, h : h + 3, w : w + 3] * params["W2"], axis=(0, 2, 3)
                )
    dZ1 = dA1 * (A1 > 0)  # derivative of ReLU
    grads["W1"] = np.zeros_like(params["W1"])  # initialize gradient for W1
    grads["b1"] = np.zeros_like(params["b1"])  # initialize gradient for b1
    for i in range(X.shape[0]):  # loop over batch
        grads["W1"] += np.sum(
            dZ1[i, :, :, :, np.newaxis] * X[i, np.newaxis, :, :, :], axis=(1, 2, 3)
        )
        grads["b1"] += np.sum(dZ1[i, :, :, :], axis=(1, 2))
    grads["W1"] /= X.shape[0]
    grads["b1"] /= X.shape[0]


def update_params(params, grads, learning_rate):
    # update parameters using gradients
    for key in params.keys():
        params[key] -= learning_rate * grads[key]


def train(num_epochs=10, batch_size=64, learning_rate=0.001):
    # get data loaders
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = BabyCNN()  # initialize model
    for epoch in range(num_epochs):
        # training phase
        for X_batch, Y_batch in train_loader:
            # forward pass
            cache = {
                "A1": model.conv_forward(
                    X_batch, model.params["W1"], model.params["b1"], pad="same"
                ),
                "A2": None,
                "A3": None,
                "A4": None,
            }
            cache["A1"] = model.relu(cache["A1"])
            cache["A2"] = model.conv_forward(
                cache["A1"], model.params["W2"], model.params["b2"], pad="same"
            )
            cache["A2"] = model.layer_norm(cache["A2"])
            cache["A2"] = model.relu(cache["A2"])
            cache["A2"] = model.max_pool(cache["A2"])
            cache["A2"] = cache["A2"].reshape((cache["A2"].shape[0], -1))  # flatten
            cache["A3"] = model.relu(
                np.dot(cache["A2"], model.params["W3"]) + model.params["b3"].T
            )
            cache["A4"] = model.softmax(
                np.dot(cache["A3"], model.params["W4"]) + model.params["b4"].T
            )

            # logging shapes
            print(f"X_batch shape: {X_batch.shape}")
            print(f"Y_batch shape: {Y_batch.shape}")
            for key in cache:
                if cache[key] is not None:
                    print(f"{key} shape: {cache[key].shape}")

            # initialize gradients
            grads = {key: np.zeros_like(value) for key, value in model.params.items()}
            # backward pass
            backpropagate(model.params, grads, cache, X_batch, Y_batch)
            # update parameters
            update_params(model.params, grads, learning_rate)

        # evaluation phase
        train_loss, train_accuracy = 0, 0
        for X_batch, Y_batch in train_loader:
            Y_hat = model.forward(X_batch)
            train_loss += cross_entropy_loss(Y_batch, Y_hat)
            train_accuracy += compute_accuracy(Y_batch, Y_hat)
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        test_loss, test_accuracy = 0, 0
        for X_batch, Y_batch in test_loader:
            Y_hat = model.forward(X_batch)
            test_loss += cross_entropy_loss(Y_batch, Y_hat)
            test_accuracy += compute_accuracy(Y_batch, Y_hat)
        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)

        # print epoch results
        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%"
        )


if __name__ == "__main__":
    train()
