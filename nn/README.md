# nn

`nn/` contains all the different neural network implementations of BabyCNN. Each folder contains a `README.md` file that describes its implementation. Training and inference is done on [MNIST](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/) and [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist).

## Framework-Agnostic Pseudocode for BabyCNN

```python
class BabyCNN:
    def __init__():
        self.conv1 = Conv2D(1, 32, kernel_size=3, stride=1)
        self.conv2 = Conv2D(32, 64, kernel_size=3, stride=1)
        self.batch_norm1 = BatchNormalization(64)
        self.fc1 = FullyConnected(input_size=12 * 12 * 64, output_size=128)
        self.dropout1 = Dropout(rate=0.5)
        self.fc2 = FullyConnected(input_size=128, output_size=10)
        
    def forward(x):
        # first convolutional layer followed by ReLU activation
        x = self.conv1(x)
        x = ReLU(x)
        
        # second convolutional layer followed by batchnorm and ReLU activation
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = ReLU(x)

        # max pooling layer with a stride of 2
        x = MaxPool2D(x, pool_size=2)
        
        # flatten the output of the max pooling layer
        x = Flatten(x)
        
        # apply the first fully connected layer to get the output of the first hidden layer
        x = self.fc1(x)
        x = self.dropout1(x)
        x = ReLU(x)
        
        # apply the second fully connected layer to get the output of the second hidden layer
        x = self.fc2(x)
        
        # softmax activation
        x = Softmax(x)
        
        return x
```

It's also helpful to go through the journey of an input image to understand what's going on.

## BabyCNN: Journey of an Input Image

This describes the journey of an input image through the BabyCNN model, including the shape transformations and intuitive understanding of each layer.

## Shape Transformations and Visualizations

### 1. Input Image: 28x28 Grayscale Image

- **Shape**: `(28, 28, 1)`
- **Description**: The input image is a 28x28 grayscale image where each pixel's intensity ranges from 0 to 1.
- **Intuition**: This is the raw input data fed into the neural network.

### 2. Conv2D Layer 1: 32 Filters of Size 3x3, Stride 1

- **Output Shape**: `(26, 26, 32)`
- **Description**: The first convolutional layer applies 32 different 3x3 filters to the input image.
- **Intuition**: This layer extracts basic features like edges and textures from the image.

### 3. ReLU Activation: Applied After Conv2D Layer 1

- **Output Shape**: `(26, 26, 32)`
- **Description**: Applies the Rectified Linear Unit (ReLU) activation function to introduce non-linearity.
- **Intuition**: Helps the model learn complex patterns by applying non-linear activation to the feature maps.

### 4. Conv2D Layer 2: 64 Filters of Size 3x3, Stride 1

- **Output Shape**: `(24, 24, 64)`
- **Description**: The second convolutional layer applies 64 different 3x3 filters to the output of the first layer.
- **Intuition**: This layer captures more complex features by combining the basic features extracted by the first layer.

### 5. Batch Normalization: Applied After Conv2D Layer 2

- **Output Shape**: `(24, 24, 64)`
- **Description**: Normalizes the feature maps to have zero mean and unit variance.
- **Intuition**: Stabilizes and speeds up the training process by normalizing the output of the convolutional layer.

### 6. ReLU Activation: Applied After Batch Normalization

- **Output Shape**: `(24, 24, 64)`
- **Description**: Applies the ReLU activation function again after batch normalization.
- **Intuition**: Maintains the non-linearity and helps in learning complex patterns.

### 7. MaxPooling2D: Pool Size 2x2

- **Output Shape**: `(12, 12, 64)`
- **Description**: Applies max pooling with a 2x2 filter and stride of 2 to reduce the spatial dimensions.
- **Intuition**: Reduces the dimensionality of the feature maps, retaining the most important features while reducing computational complexity.

### 8. Flatten: Flatten the Output

- **Output Shape**: `(9216)`
- **Description**: Flattens the 3D feature maps into a 1D vector.
- **Intuition**: Prepares the data for the fully connected layers by converting the 2D feature maps into a 1D vector.

### 9. Fully Connected Layer 1: 128 Neurons

- **Output Shape**: `(128)`
- **Description**: Applies a fully connected layer with 128 neurons to the flattened vector.
- **Intuition**: Acts as the brain of the network where most of the learning happens, transforming the high-dimensional data into 128 features.

### 10. Dropout: Applied After Fully Connected Layer 1

- **Output Shape**: `(128)`
- **Description**: Randomly sets a fraction of the input units to 0 at each update during training time to prevent overfitting.
- **Intuition**: Improves generalization by preventing the model from relying too much on specific neurons.

### 11. ReLU Activation: Applied After Dropout

- **Output Shape**: `(128)`
- **Description**: Applies the ReLU activation function again after dropout.
- **Intuition**: Maintains non-linearity and ensures only active neurons are contributing to the output.

### 12. Fully Connected Layer 2: 10 Neurons

- **Output Shape**: `(10)`
- **Description**: Applies a fully connected layer with 10 neurons to the output of the previous layer.
- **Intuition**: Maps the high-level features to the output classes.

### 13. Softmax Activation: Applied to Get Probabilities

- **Output Shape**: `(10)`
- **Description**: Applies the softmax activation function to get the probability distribution over the 10 classes.
- **Intuition**: Converts the logits into probabilities, allowing for the final classification decision.

## Frameworks

Within each folder, you'll find a README that describes the implementation, alongside a `train.py` script for training the model, and a `infer.py` script for inference.

- `/pytorch`: PyTorch implementation of BabyCNN.
- `/jax`: JAX implementation of BabyCNN.
- `/tf2`: TensorFlow 2 implementation of BabyCNN.
- `/c`: C implementation of BabyCNN, using a CPU.
- `/cuda`: CUDA implementation of BabyCNN, using C++ and a GPU.
- `/cutlass`: CUDA implementation of BabyCNN, using C++ and CUTLASS (a kernel template library built on CUDA).
- `/triton`: Triton implementation of BabyCNN, using Python and a GPU.
- `/mlx`: MLX implementation of BabyCNN, using Apple Silicon.
