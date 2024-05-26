# jax Implementation of BabyCNN

## Overview

This folder contains the implementation of BabyCNN using jax.

## Setup Instructions

### Dependencies

- jax
- flax
- tensorflow-datasets
- optax

### Setup

For CPU-only training:

```sh
pip install "jax[cpu]" flax tensorflow-datasets optax
```

For GPU-accelerated training:

```sh
pip install "jax[cuda12]" flax tensorflow-datasets optax
```

## Training

Run the following command to train the model:

```sh
python train.py
```
