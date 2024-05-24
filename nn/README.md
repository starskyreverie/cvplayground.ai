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

## Journey of an Input Image

### shape transformations

1. **input image**: 28x28 grayscale image.
   - **shape**: (28, 28, 1)
   - **example**: `[[0.5, 0.3, ...], [0.2, 0.6, ...], ...]`
   - **meaning**: raw pixel values of the grayscale image.

2. **conv2d layer 1**: 32 filters of size 3x3, stride 1.
   - **output shape**: (26, 26, 32)
   - **example**: `[[[0.1, ...], ...], [[0.2, ...], ...], ...]`
   - **meaning**: feature maps extracted by the first convolutional layer.

3. **relu activation**: applied after conv2d layer 1.
   - **output shape**: (26, 26, 32)
   - **example**: `[[[0.1, ...], ...], [[0.2, ...], ...], ...]`
   - **meaning**: feature maps with non-linear activation applied.

4. **conv2d layer 2**: 64 filters of size 3x3, stride 1.
   - **output shape**: (24, 24, 64)
   - **example**: `[[[0.2, ...], ...], [[0.3, ...], ...], ...]`
   - **meaning**: feature maps extracted by the second convolutional layer.

5. **batch normalization**: applied after conv2d layer 2.
   - **output shape**: (24, 24, 64)
   - **example**: `[[[0.1, ...], ...], [[0.2, ...], ...], ...]`
   - **meaning**: normalized feature maps.

6. **relu activation**: applied after batch normalization.
   - **output shape**: (24, 24, 64)
   - **example**: `[[[0.1, ...], ...], [[0.2, ...], ...], ...]`
   - **meaning**: normalized feature maps with non-linear activation applied.

7. **maxpooling2d**: pool size 2x2.
   - **output shape**: (12, 12, 64)
   - **example**: `[[[0.3, ...], ...], [[0.4, ...], ...], ...]`
   - **meaning**: downsampled feature maps.

8. **flatten**: flatten the output.
   - **output shape**: (9216)
   - **example**: `[0.3, 0.4, ...]`
   - **meaning**: flattened feature maps.

9. **fully connected layer 1**: 128 neurons.
   - **output shape**: (128)
   - **example**: `[0.1, 0.2, ...]`
   - **meaning**: high-level features.

10. **dropout**: applied after fully connected layer 1.
    - **output shape**: (128)
    - **example**: `[0.1, 0.0, ...]`
    - **meaning**: high-level features with dropout applied.

11. **relu activation**: applied after dropout.
    - **output shape**: (128)
    - **example**: `[0.1, 0.2, ...]`
    - **meaning**: high-level features with non-linear activation applied.

12. **fully connected layer 2**: 10 neurons.
    - **output shape**: (10)
    - **example**: `[0.1, 0.3, ...]`
    - **meaning**: logits for each class.

13. **softmax activation**: applied to get probabilities.
    - **output shape**: (10)
    - **example**: `[0.1, 0.7, ...]`
    - **meaning**: probability distribution over 10 classes.
