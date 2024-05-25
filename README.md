# CVPlayground

[cvplayground.ai](https://cvplayground.ai) is a place to play with computer vision models. It's currently a work in progress.

## Overview

CVPlayground is designed to explore various methods of defining, training, and inferring computer vision models. The core motivation is to experiment with different frameworks and hardware implementations with BabyCNN.

Check out the README in `nn/` to learn more about `BabyCNN` and the different ways we implement it.

## Motivation

The project aims to implement BabyCNN using various methods:

- High-level frameworks like PyTorch, JAX, and TensorFlow.
- Low-level implementations in C and CUDA.
- Custom CUDA kernels.
- Advanced optimization techniques using Triton and CUTLASS.
- Execution on different hardware including ASICs, GPUs, and ML accelerators like MLX.

## Future Work

In the future, I have plans to extend the project to work at the scale of [DINOv2](https://arxiv.org/pdf/2304.07193), but this is very non-trivial.
