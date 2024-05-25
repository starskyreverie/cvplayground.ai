# Distributed Training

This repository contains examples of distributed training methods using PyTorch. We explore three different distributed training strategies:

1. **Distributed Data Parallel (DDP)**
2. **Fully Sharded Data Parallel (FSDP)**
3. **Pipeline Parallelism (PiPPy)**

## 1. Distributed Data Parallel (DDP)

### DDP Overview

DDP replicates the entire model on each GPU and synchronizes gradients during the backward pass. It scales well across multiple GPUs, making it efficient for large models.

**Example with BabyCNN and 4 GPUs:**

- **Model Replication:** Each GPU (GPU 0, GPU 1, GPU 2, GPU 3) gets a complete copy of BabyCNN.
- **Forward Pass:** Each GPU computes the forward pass independently with its mini-batch of data.
  - Suppose each GPU processes a mini-batch of size 16, resulting in an input tensor shape of `(16, 1, 28, 28)`.
  - After `conv1`: `(16, 32, 26, 26)`
  - After `conv2`: `(16, 64, 24, 24)`
  - After `layer_norm1`: `(16, 64, 24, 24)`
  - After `max_pool1`: `(16, 64, 12, 12)`
  - After `flatten`: `(16, 9216)`
  - After `fc1`: `(16, 128)`
  - After `fc2`: `(16, 10)`
- **Backward Pass:** Gradients from each GPU are synchronized and averaged. This ensures all model replicas are updated consistently.

### How DDP Works

- **Initialization:** `dist.init_process_group("nccl")` sets up the communication backend.
- **Model Wrapping:** `model = DDP(model, device_ids=[rank])` wraps the model for distributed training.
- **Gradient Synchronization:** Gradients are averaged across all GPUs during the backward pass.

### DDP Code Snippet

- Initialize process group: `dist.init_process_group("nccl")`
- Wrap model: `model = DDP(model, device_ids=[rank])`
- Example script: `train_ddp.py` (use `run_ddp.sh` to run)

## 2. Fully Sharded Data Parallel (FSDP)

### FSDP Overview

FSDP reduces memory usage by sharding model parameters, gradients, and optimizer states across GPUs. It's useful for very large models that can't fit into a single GPU's memory.

**Example with BabyCNN and 4 GPUs:**

- **Parameter Sharding:** Model parameters are split and distributed across GPU 0, GPU 1, GPU 2, GPU 3.
  - Suppose the model has 10 million parameters. Each GPU holds 2.5 million parameters.
- **Forward Pass:** Each GPU computes the forward pass with its shard of parameters.
  - Input tensor on each GPU: `(16, 1, 28, 28)`
  - Sharded computations occur, but intermediate tensor shapes remain the same as DDP.
- **Backward Pass:** Gradients are also sharded and synchronized in a way that minimizes memory usage.

### How FSDP Works

- **Initialization:** `dist.init_process_group("nccl")` sets up the communication backend.
- **Model Wrapping:** `model = FSDP(model, auto_wrap_policy=default_auto_wrap_policy)` wraps the model with FSDP for sharding.
- **Gradient and Parameter Sharding:** Both are distributed across GPUs to reduce memory overhead.

### FSDP Code Snippet

- Wrap model: `model = FSDP(model, auto_wrap_policy=default_auto_wrap_policy)`
- Example script: `train_fsdp.py` (use `run_fsdp.sh` to run)

## 3. Pipeline Parallelism (PiPPy)

### PiPPy Overview

Pipeline parallelism splits the model into multiple stages, each on a different GPU. It's useful for models too large for one GPU but can be split and executed in sequence.

**Example with BabyCNN and 4 GPUs:**

- **Model Partitioning:** The model is divided into four stages, each placed on GPU 0, GPU 1, GPU 2, GPU 3.
  - **Stage 1 (GPU 0):** `conv1`
  - **Stage 2 (GPU 1):** `conv2`
  - **Stage 3 (GPU 2):** `layer_norm1` + `max_pool1`
  - **Stage 4 (GPU 3):** `flatten` + `fc1` + `fc2`
- **Pipeline Execution:** Micro-batches of data are passed through the pipeline stages sequentially.
  - Batch 1 flows from GPU 0 to GPU 3, followed by Batch 2, and so on.

### How PiPPy Works

- **Initialization:** `init_process_group(backend='nccl')` sets up the communication backend.
- **Model Wrapping:** `model = Pipe(model, chunks=8)` wraps the model for pipeline parallelism.
- **Sequential Execution:** Micro-batches pass through each stage in sequence, enabling parallelism across GPUs.

### PiPPy Code Snippet

- Wrap model: `model = Pipe(model, chunks=8)`
- Example script: `train_pippy.py` (use `run_pippy.sh` to run)