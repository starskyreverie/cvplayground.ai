import numpy as np


def initialize_weights(shape):
    return np.random.randn(*shape) * 0.01


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(predictions, labels):
    num_samples = predictions.shape[0]
    log_probs = -np.log(predictions[range(num_samples), labels])
    loss = np.sum(log_probs) / num_samples
    return loss


def log_shape(name, tensor):
    print(f"{name} shape: {tensor.shape}")


class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.stride = stride
        self.filters = initialize_weights(
            (out_channels, in_channels, kernel_size, kernel_size)
        )

    def forward(self, input):
        """
        Forward pass for ConvLayer
        input: shape (batch_size, in_channels, input_height, input_width)
        output: shape (batch_size, out_channels, output_height, output_width)

        Example:
        input shape: (5, 1, 28, 28)
        filters shape: (32, 1, 3, 3)
        output shape: (5, 32, 26, 26)
        """
        self.last_input = input
        batch_size, in_channels, input_height, input_width = input.shape
        out_channels, _, kernel_size, _ = self.filters.shape
        output_height = (input_height - kernel_size) // self.stride + 1
        output_width = (input_width - kernel_size) // self.stride + 1
        output = np.zeros((batch_size, out_channels, output_height, output_width))

        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(in_channels):
                    for m in range(0, input_height - kernel_size + 1, self.stride):
                        for n in range(0, input_width - kernel_size + 1, self.stride):
                            region = input[
                                i, k, m : m + kernel_size, n : n + kernel_size
                            ]
                            output[i, j, m // self.stride, n // self.stride] += np.sum(
                                region * self.filters[j, k]
                            )

        log_shape("conv output", output)
        return output

    def backward(self, d_output, learning_rate):
        """
        Backward pass for ConvLayer
        d_output: shape (batch_size, out_channels, output_height, output_width)
        d_filters: shape (out_channels, in_channels, kernel_size, kernel_size)
        d_input: shape (batch_size, in_channels, input_height, input_width)

        Example:
        d_output shape: (5, 32, 26, 26)
        d_filters shape: (32, 1, 3, 3)
        d_input shape: (5, 1, 28, 28)
        """
        d_filters = np.zeros(self.filters.shape)
        batch_size, in_channels, input_height, input_width = self.last_input.shape
        out_channels, _, kernel_size, _ = self.filters.shape

        d_input = np.zeros(self.last_input.shape)

        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(in_channels):
                    for m in range(0, input_height - kernel_size + 1, self.stride):
                        for n in range(0, input_width - kernel_size + 1, self.stride):
                            region = self.last_input[
                                i, k, m : m + kernel_size, n : n + kernel_size
                            ]
                            d_filters[j, k] += (
                                d_output[i, j, m // self.stride, n // self.stride]
                                * region
                            )
                            d_input[i, k, m : m + kernel_size, n : n + kernel_size] += (
                                d_output[i, j, m // self.stride, n // self.stride]
                                * self.filters[j, k]
                            )

        self.filters -= learning_rate * d_filters
        return d_input


class MaxPoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        """
        Forward pass for MaxPoolLayer
        input: shape (batch_size, in_channels, input_height, input_width)
        output: shape (batch_size, in_channels, output_height, output_width)

        Example:
        input shape: (5, 32, 26, 26)
        output shape: (5, 32, 13, 13)
        """
        self.last_input = input
        batch_size, in_channels, input_height, input_width = input.shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, in_channels, output_height, output_width))

        for i in range(batch_size):
            for j in range(in_channels):
                for m in range(0, input_height - self.pool_size + 1, self.stride):
                    for n in range(0, input_width - self.pool_size + 1, self.stride):
                        region = input[
                            i, j, m : m + self.pool_size, n : n + self.pool_size
                        ]
                        output[i, j, m // self.stride, n // self.stride] = np.max(
                            region
                        )

        log_shape("maxpool output", output)
        return output

    def backward(self, d_output):
        """
        Backward pass for MaxPoolLayer
        d_output: shape (batch_size, in_channels, output_height, output_width)
        d_input: shape (batch_size, in_channels, input_height, input_width)

        Example:
        d_output shape: (5, 32, 13, 13)
        d_input shape: (5, 32, 26, 26)
        """
        d_input = np.zeros(self.last_input.shape)
        batch_size, in_channels, input_height, input_width = self.last_input.shape

        for i in range(batch_size):
            for j in range(in_channels):
                for m in range(0, input_height - self.pool_size + 1, self.stride):
                    for n in range(0, input_width - self.pool_size + 1, self.stride):
                        region = self.last_input[
                            i, j, m : m + self.pool_size, n : n + self.pool_size
                        ]
                        max_val = np.max(region)
                        for r in range(self.pool_size):
                            for c in range(self.pool_size):
                                if region[r, c] == max_val:
                                    d_input[i, j, m + r, n + c] = d_output[
                                        i, j, m // self.stride, n // self.stride
                                    ]

        return d_input


class LayerNorm:
    def __init__(self, normalized_shape):
        self.gamma = np.ones((1, *normalized_shape))
        self.beta = np.zeros((1, *normalized_shape))

    def forward(self, input):
        """
        Forward pass for LayerNorm
        input: shape (batch_size, in_channels, height, width)
        output: same shape as input

        Example:
        input shape: (5, 64, 24, 24)
        """
        self.last_input = input
        self.mean = np.mean(input, axis=(1, 2, 3), keepdims=True)
        self.variance = np.var(input, axis=(1, 2, 3), keepdims=True)
        self.normalized = (input - self.mean) / np.sqrt(self.variance + 1e-5)
        output = self.gamma * self.normalized + self.beta
        log_shape("layer norm output", output)
        return output

    def backward(self, d_output, learning_rate):
        """
        Backward pass for LayerNorm
        d_output: same shape as input
        """
        d_gamma = np.sum(d_output * self.normalized, axis=(0, 2, 3), keepdims=True)
        d_beta = np.sum(d_output, axis=(0, 2, 3), keepdims=True)
        d_normalized = d_output * self.gamma

        d_variance = np.sum(
            d_normalized
            * (self.last_input - self.mean)
            * -0.5
            * np.power(self.variance + 1e-5, -1.5),
            axis=(1, 2, 3),
            keepdims=True,
        )
        d_mean = np.sum(
            d_normalized * -1.0 / np.sqrt(self.variance + 1e-5),
            axis=(1, 2, 3),
            keepdims=True,
        ) + d_variance * np.mean(
            -2.0 * (self.last_input - self.mean), axis=(1, 2, 3), keepdims=True
        )

        d_input = (
            (d_normalized / np.sqrt(self.variance + 1e-5))
            + (
                d_variance
                * 2.0
                * (self.last_input - self.mean)
                / self.last_input.shape[0]
            )
            + (d_mean / self.last_input.shape[0])
        )

        self.gamma -= learning_rate * d_gamma
        self.beta -= learning_rate * d_beta

        return d_input


class Dropout:
    def __init__(self, drop_prob=0.5):
        self.drop_prob = drop_prob

    def forward(self, input, train=True):
        """
        Forward pass for Dropout
        input: any shape
        output: same shape as input

        Example:
        input shape: (5, 128)
        """
        if train:
            self.mask = (np.random.rand(*input.shape) > self.drop_prob) / (
                1.0 - self.drop_prob
            )
            return input * self.mask
        else:
            return input

    def backward(self, d_output):
        """
        Backward pass for Dropout
        d_output: same shape as input
        """
        return d_output * self.mask


class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = initialize_weights((input_size, output_size))
        self.biases = initialize_weights((output_size,))

    def forward(self, input):
        """
        Forward pass for FullyConnectedLayer
        input: shape (batch_size, input_size)
        output: shape (batch_size, output_size)

        Example:
        input shape: (5, 12*12*64)
        output shape: (5, 128)
        """
        self.last_input_shape = input.shape
        self.last_input = input.reshape(input.shape[0], -1)
        output = np.dot(self.last_input, self.weights) + self.biases
        log_shape("fc output", output)
        return output

    def backward(self, d_output, learning_rate):
        """
        Backward pass for FullyConnectedLayer
        d_output: shape (batch_size, output_size)
        d_weights: shape (input_size, output_size)
        d_biases: shape (output_size,)
        d_input: shape (batch_size, input_size)

        Example:
        d_output shape: (5, 128)
        d_weights shape: (12*12*64, 128)
        d_input shape: (5, 12*12*64)
        """
        d_weights = np.dot(self.last_input.T, d_output)
        d_biases = np.sum(d_output, axis=0)
        d_input = np.dot(d_output, self.weights.T).reshape(self.last_input_shape)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input


class BabyCNN:
    def __init__(self):
        self.conv1 = ConvLayer(1, 32, 3, stride=1)
        self.conv2 = ConvLayer(32, 64, 3, stride=1)
        self.layer_norm1 = LayerNorm([64, 24, 24])
        self.max_pool1 = MaxPoolLayer(2, stride=2)
        self.fc1 = FullyConnectedLayer(12 * 12 * 64, 128)
        self.dropout1 = Dropout(drop_prob=0.5)
        self.fc2 = FullyConnectedLayer(128, 10)

    def forward(self, x):
        log_shape("input", x)
        x = self.conv1.forward(x)
        x = relu(x)
        log_shape("after relu 1", x)
        x = self.conv2.forward(x)
        x = self.layer_norm1.forward(x)
        x = relu(x)
        log_shape("after relu 2", x)
        x = self.max_pool1.forward(x)
        self.pool_output_shape = x.shape  # store shape for backward pass
        x = x.reshape(
            x.shape[0], -1
        )  # flatten the output for the fully connected layer
        x = self.fc1.forward(x)
        x = self.dropout1.forward(x, train=True)
        x = relu(x)
        log_shape("after relu 3", x)
        x = self.fc2.forward(x)
        x = softmax(x)
        log_shape("after softmax", x)
        return x

    def backward(self, d_output, learning_rate):
        d_output = self.fc2.backward(d_output, learning_rate)
        d_output = relu_derivative(d_output)
        d_output = self.dropout1.backward(d_output)
        d_output = self.fc1.backward(d_output, learning_rate)
        d_output = d_output.reshape(
            self.pool_output_shape
        )  # reshape to match max pool output
        d_output = self.max_pool1.backward(d_output)
        d_output = relu_derivative(d_output)
        d_output = self.layer_norm1.backward(d_output, learning_rate)
        d_output = self.conv2.backward(d_output, learning_rate)
        d_output = relu_derivative(d_output)
        d_output = self.conv1.backward(d_output, learning_rate)


def main():
    """
    This is just for testing. Obviously, we aren't actually learning anything,
    since we're using dummy inputs and labels.
    """
    np.random.seed(0)
    batch_size = 5
    dummy_input = np.random.randn(batch_size, 1, 28, 28)
    dummy_labels = np.random.randint(0, 10, batch_size)

    model = BabyCNN()
    learning_rate = 0.01
    num_iterations = 3

    for iteration in range(num_iterations):
        output = model.forward(dummy_input)
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print("model output:", output)

        loss = cross_entropy_loss(output, dummy_labels)
        print("loss:", loss)

        d_output = output
        d_output[range(batch_size), dummy_labels] -= 1
        d_output /= batch_size
        model.backward(d_output, learning_rate)
        print("updated model parameters")


if __name__ == "__main__":
    main()
