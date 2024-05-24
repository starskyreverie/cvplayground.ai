# CVPlayground

[cvplayground.ai](https://cvplayground.ai) is a place to play with computer vision models. It's currently a work in progress.

## Overview

CVPlayground is designed to explore various methods of defining, training, and inferring computer vision models. The core motivation is to experiment with different frameworks and hardware implementations, starting with a sufficiently complex architecture called BabyCNN.

### BabyCNN Example

BabyCNN is a convolutional neural network designed to work with the MNIST dataset. This network serves as a baseline to explore different methods of execution.

For example, we can define the network in PyTorch as follows:

```python
import torch
import torch.nn as nn

class BabyCNN(nn.Module):
    def __init__(self):
        super(BabyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = x.view(-1, 12 * 12 * 64)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return nn.LogSoftmax(dim=1)(x)
```

## Motivation

The project aims to implement BabyCNN using various methods:

- High-level frameworks like PyTorch, JAX, and TensorFlow.
- Low-level implementations in C and CUDA.
- Custom CUDA kernels.
- Advanced optimization techniques using Triton and CUTLASS.
- Execution on different hardware including ASICs, GPUs, and ML accelerators like MLX.
