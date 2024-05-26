# nn

`nn/` contains all the different neural network implementations of BabyCNN. Each folder contains a `README.md` file that describes its implementation. Training and inference is done on [MNIST](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/).

You can install the MNIST dataset by running `python install_mnist.py` and test the dataset by running `python load_mnist.py`.

## Framework-Agnostic Pseudocode for BabyCNN

```python
class BabyCNN:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1)  # Input: (28, 28, 1) -> Output: (26, 26, 32)
        self.conv2 = Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1)  # Input: (26, 26, 32) -> Output: (24, 24, 64)
        self.layer_norm1 = LayerNormalization(num_features=64)  # Input: (24, 24, 64) -> Output: (24, 24, 64)
        self.fc1 = FullyConnected(input_size=12 * 12 * 64, output_size=128)  # Input: (9216) -> Output: (128)
        self.dropout1 = Dropout(rate=0.5)  # Input: (128) -> Output: (128)
        self.fc2 = FullyConnected(input_size=128, output_size=10)  # Input: (128) -> Output: (10)
        
    def forward(self, x):
        x = self.conv1(x)  # (28, 28, 1) -> (26, 26, 32)
        x = ReLU(x)  # (26, 26, 32) -> (26, 26, 32)
        x = self.conv2(x)  # (26, 26, 32) -> (24, 24, 64)
        x = self.layer_norm1(x)  # (24, 24, 64) -> (24, 24, 64)
        x = ReLU(x)  # (24, 24, 64) -> (24, 24, 64)
        x = MaxPool2D(x, pool_size=2, stride=2)  # (24, 24, 64) -> (12, 12, 64)
        x = Flatten(x)  # (12, 12, 64) -> (9216)
        x = self.fc1(x)  # (9216) -> (128)
        x = self.dropout1(x)  # (128) -> (128)
        x = ReLU(x)  # (128) -> (128)
        x = self.fc2(x)  # (128) -> (10)
        x = Softmax(x)  # (10) -> (10)
        
        return x

```

It's also helpful to go through the journey of an input image to understand what's going on.

## BabyCNN: Journey of an Input Image

This describes the journey of an input image through the BabyCNN model, including the shape transformations and intuitive understanding of each layer.

## Shape Transformations and Visualizations

### 1. Input Image: 28x28 Grayscale Image

- **Shape**: `(28, 28, 1)`
- **Description**: The input image is a 28x28 grayscale image where each pixel's intensity ranges from 0 to 1.
- **Intuition**: This is the raw input data fed into the neural network. The shape `(28, 28, 1)` means the image is 28 pixels high, 28 pixels wide, and has 1 color channel (grayscale).

### 2. Conv2D Layer 1: 32 Filters of Size 3x3, Stride 1

- **Output Shape**: `(26, 26, 32)`
- **Description**: The first convolutional layer applies 32 different 3x3 filters to the input image.
- **Intuition**: This layer extracts basic features like edges and textures from the image.

#### Detailed Breakdown:

- **Filter Size**: Each filter is `3x3`, meaning it covers a 3x3 region of the input image.
- **Stride**: A stride of 1 means the filter moves one pixel at a time across the image.
- **Output Height/Width Calculation**:
  - Formula: `(Input Size - Filter Size) / Stride + 1`
  - For height and width: `(28 - 3) / 1 + 1 = 26`
- **Output Depth**: The depth of the output is equal to the number of filters used, which is 32. Each filter captures different features, leading to 32 feature maps.
- **Why 32 Filters**: Using 32 filters allows the network to capture a diverse set of basic features from the input image. Increasing the number of filters generally allows the network to learn more complex features, but it also increases the computational cost and risk of overfitting.

### 3. ReLU Activation: Applied After Conv2D Layer 1

- **Output Shape**: `(26, 26, 32)`
- **Description**: Applies the Rectified Linear Unit (ReLU) activation function to introduce non-linearity.
- **Intuition**: ReLU replaces all negative values with zero. Non-linearity helps because it allows the network to learn from and model more complex patterns. If we only used linear transformations, no matter how many layers we add, the final output would still be a linear function of the input. ReLU introduces the necessary non-linearity to ensure the neural network can model a wider range of functions.

### 4. Conv2D Layer 2: 64 Filters of Size 3x3, Stride 1

- **Output Shape**: `(24, 24, 64)`
- **Description**: The second convolutional layer applies 64 different 3x3 filters to the output of the first layer.
- **Intuition**: This layer captures more complex features by combining the basic features extracted by the first layer.

#### Detailed Breakdown:

- **Filter Size**: Each filter is `3x3`.
- **Stride**: A stride of 1.
- **Output Height/Width Calculation**:
  - Formula: `(Input Size - Filter Size) / Stride + 1`
  - For height and width: `(26 - 3) / 1 + 1 = 24`
- **Output Depth**: The depth of the output is equal to the number of filters used, which is 64.
- **Why 64 Filters**: Increasing the number of filters to 64 allows the network to learn even more complex and abstract features by building on the simpler features captured by the first layer. This progression helps the network to better understand the input data.

### 5. Layer Normalization: Applied After Conv2D Layer 2

- **Output Shape**: `(24, 24, 64)`
- **Description**: Normalizes the feature maps to have zero mean and unit variance.
- **Intuition**: Stabilizes and speeds up the training process by normalizing the output of the convolutional layer. Layer normalization helps to maintain the distributions of the feature maps, reducing the internal covariate shift, and making the network less sensitive to the initial weights.

### 6. ReLU Activation: Applied After Layer Normalization

- **Output Shape**: `(24, 24, 64)`
- **Description**: Applies the ReLU activation function again after layer normalization.
- **Intuition**: Maintains non-linearity and prevents the model from being a simple linear transformation. The non-linearity introduced by ReLU after layer normalization ensures that the model can learn complex functions and dependencies in the data.

### 7. MaxPooling2D: Pool Size 2x2

- **Output Shape**: `(12, 12, 64)`
- **Description**: Applies max pooling with a 2x2 filter and stride of 2 to reduce the spatial dimensions.
- **Intuition**: Reduces the dimensionality of the feature maps, retaining the most important features while reducing computational complexity. Max pooling downsamples the input by taking the maximum value in each 2x2 patch, effectively summarizing the presence of features in that region.

#### Detailed Breakdown:

- **Pool Size**: Each pooling operation is `2x2`, meaning it covers a 2x2 region.
- **Stride**: A stride of 2.
- **Output Height/Width Calculation**:
  - Formula: `(Input Size - Pool Size) / Stride + 1`
  - For height and width: `(24 - 2) / 2 + 1 = 12`

### 8. Flatten: Flatten the Output

- **Output Shape**: `(9216)`
- **Description**: Flattens the 3D feature maps into a 1D vector.
- **Intuition**: Prepares the data for the fully connected layers by converting the 2D feature maps into a 1D vector.

#### Detailed Breakdown:

- **Flatten Operation**: Converts the shape `(12, 12, 64)` into a single dimension: `12 * 12 * 64 = 9216`.

### 9. Fully Connected Layer 1: 128 Neurons

- **Output Shape**: `(128)`
- **Description**: Applies a fully connected layer with 128 neurons to the flattened vector.
- **Intuition**: Acts as the brain of the network where most of the learning happens, transforming the high-dimensional data into 128 features. Fully connected layers allow the network to learn complex representations and interactions between the features.
- **Why 128 Neurons**: Using 128 neurons strikes a balance between model complexity and computational efficiency. Increasing the number of neurons allows the network to learn more detailed patterns, but too many neurons can lead to overfitting and increased computational cost.

### 10. Dropout: Applied After Fully Connected Layer 1

- **Output Shape**: `(128)`
- **Description**: Randomly sets a fraction of the input units to 0 at each update during training time to prevent overfitting.
- **Intuition**: Improves generalization by preventing the model from relying too much on specific neurons. Dropout forces the network to learn redundant representations, making it more robust and less likely to overfit the training data.

### 11. ReLU Activation: Applied After Dropout

- **Output Shape**: `(128)`
- **Description**: Applies the ReLU activation function again after dropout.
- **Intuition**: Maintains non-linearity and ensures only active neurons contribute to the output. The non-linearity introduced by ReLU helps the model to capture more complex patterns in the data.

### 12. Fully Connected Layer 2: 10 Neurons

- **Output Shape**: `(10)`
- **Description**: Applies a fully connected layer with 10 neurons to the output of the previous layer.
- **Intuition**: Maps the high-level features to the output classes. Each neuron in this layer corresponds to one of the 10 output classes (digits 0-9).
- **Why 10 Neurons**: The number of neurons in this layer matches the number of classes in the classification problem (digits 0-9), allowing the network to output a score for each class.

### 13. Softmax Activation: Applied to Get Probabilities

- **Output Shape**: `(10)`
- **Description**: Applies the softmax activation function to get the probability distribution over the 10 classes.
- **Intuition**: Converts the logits into probabilities, allowing for the final classification decision. Softmax ensures that the output values are between 0 and 1 and sum to 100%, representing the confidence of the model in each class.

## Frameworks

Within each folder, you'll find a README that describes the implementation, alongside a `train.py` script for training the model.

### Using popular frameworks (i.e., not by hand)

- `/pytorch`: PyTorch implementation of BabyCNN.
- `/jax`: JAX implementation of BabyCNN.
- `/tf2`: TensorFlow 2 implementation of BabyCNN.
- `/mlx`: MLX implementation of BabyCNN, using Apple Silicon.

### Implementations by hand, without popular frameworks

- `/numpy`: NumPy implementation of BabyCNN, lots of stuff from scratch here. I recommend reading this implementation before 
- `/cuda`: CUDA implementation of BabyCNN, using C++ and a GPU.
- `/c`: C implementation of BabyCNN, using a CPU.
- `/cutlass`: CUDA implementation of BabyCNN, using C++ and CUTLASS (a kernel template library built on CUDA).
- `/triton`: Triton implementation of BabyCNN, using Python and a GPU.
