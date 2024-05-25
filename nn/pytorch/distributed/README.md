# Distributed Training

This repository contains examples of distributed training methods using PyTorch. We explore three different distributed training strategies:

1. **Distributed Data Parallel (DDP)**
2. **Fully Sharded Data Parallel (FSDP)**
3. **Pipeline Parallelism (PiPPy)**

## 1. Distributed Data Parallel (DDP)

### DDP Intuition

DDP replicates the entire model on each GPU and synchronizes gradients during the backward pass. It's efficient and scales well across multiple GPUs, making it ideal for large models.

**Example with BabyCNN and 4 GPUs:**

- **Model Replication:** Each GPU (GPU 0, GPU 1, GPU 2, GPU 3) gets a complete copy of BabyCNN.
- **Forward Pass:** Each GPU computes the forward pass independently with its mini-batch of data.
- **Backward Pass:** Gradients from each GPU are synchronized and averaged. This ensures all model replicas are updated consistently.

### How DDP Works

- **Each GPU has a complete copy of the model.**
- **Forward pass:** Each GPU computes loss independently.
- **Backward pass:** Gradients are synchronized across GPUs, updating model parameters consistently.

### DDP Code Differences

- Initialize process group: `dist.init_process_group("nccl")`
- Wrap model: `model = DDP(model, device_ids=[rank])`
- Example script: `train_ddp.py` (use `run_ddp.sh` to run)

## 2. Fully Sharded Data Parallel (FSDP)

### FSDP Intuition

FSDP reduces memory usage by sharding model parameters, gradients, and optimizer states across GPUs. This method is particularly useful when training very large models that don't fit into the memory of a single GPU.

**Example with BabyCNN and 4 GPUs:**

- **Parameter Sharding:** Model parameters are split and distributed across GPU 0, GPU 1, GPU 2, GPU 3.
- **Forward Pass:** Each GPU computes the forward pass with its shard of parameters.
- **Backward Pass:** Gradients are also sharded and synchronized in a way that minimizes memory usage.

### How FSDP Works

- **Model parameters are sharded across GPUs.**
- **Forward pass:** Each GPU computes with its shard of the parameters.
- **Backward pass:** Gradients are also sharded, minimizing memory usage.

### FSDP Code Differences

- Wrap model: `model = FSDP(model, auto_wrap_policy=default_auto_wrap_policy)`
- Example script: `train_fsdp.py` (use `run_fsdp.sh` to run)

## 3. Pipeline Parallelism (PiPPy)

### PiPPy Intuition

Pipeline parallelism splits the model into multiple stages, each of which can be executed on a different GPU. This method is useful when a single model is too large to fit into one GPU's memory, but the layers can be split and executed in sequence.

**Example with BabyCNN and 4 GPUs:**

- **Model Partitioning:** The model is divided into four stages, each placed on GPU 0, GPU 1, GPU 2, GPU 3.
- **Pipeline Execution:** Micro-batches of data are passed through the pipeline stages sequentially, allowing different stages to process different micro-batches in parallel.

### How PiPPy Works

- **Model is divided into pipeline stages.**
- **Each stage is placed on a different GPU.**
- **Micro-batches are passed through stages in sequence.**

### PiPPy Code Differences

- Wrap model: `model = Pipe(model, chunks=8)`
- Example script: `train_pippy.py` (use `run_pippy.sh` to run)
