# pytorch Implementation of BabyCNN

## Overview

This folder contains the implementation of BabyCNN using pytorch. Besides the main scripts, there's also `distributed/`, and `/compression` folders that contain scripts for distributed training and compression (pruning and quantization).

## Setup Instructions

### Dependencies

- torch
- torchvision

### Setup

```sh
pip install torch torchvision
```

## Training

Run the following command to train the model:

```sh
python train.py
```
