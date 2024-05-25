# cutlass Implementation of BabyCNN

## Overview

This folder contains the implementation of BabyCNN using cutlass.

## Setup Instructions

### Dependencies

- CUTLASS
- nvcc (CUDA Toolkit)

### Setup

```sh
export CUTLASS_PATH=/path/to/cutlass
nvcc -I$CUTLASS_PATH/include -I$CUTLASS_PATH/tools/util/include -o train train.cu
nvcc -I$CUTLASS_PATH/include -I$CUTLASS_PATH/tools/util/include -o infer infer.cu
```

## Training

Run the following command to train the model:

```sh
python train.cu
```
