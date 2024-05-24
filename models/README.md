# Models

`models` contains all the different implementations of BabyCNN. Each folder contains a `README.md` file that describes its implementation.

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

Training is generally done on MNIST.
