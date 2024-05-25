# Model Compression

This repository contains examples of model compression methods using PyTorch. We explore two main types of compression:

1. **Pruning**
2. **Quantization**

## 1. Pruning

### Pruning Overview

Pruning involves removing less important weights from the model, reducing its size and potentially improving inference speed. The goal is to make the model more efficient without significantly affecting its accuracy.

**Example with BabyCNN and Pruning:**

- **Model Preparation:** Define the model.
- **Apply Pruning:** Prune less important weights.
- **Training:** Train the pruned model to fine-tune its performance.

### How Pruning Works

1. **Model Preparation:**
   - Define the model as usual. For instance, BabyCNN with layers such as `conv1`, `conv2`, etc.
   - Initialize the model weights normally.

2. **Apply Pruning:**
   - **Unstructured Pruning:** This method zeroes out individual weights. For example, before pruning, the weights in `conv1` might look like this:

     ```python
     tensor([[ 0.1, -0.2,  0.3], [ 0.4,  0.5, -0.6], [-0.7,  0.8,  0.9]])
     ```

     After applying `prune.l1_unstructured(module, name='weight', amount=0.4)`, 40% of the smallest magnitude weights are set to zero:

     ```python
     tensor([[ 0.1, -0.2,  0.0], [ 0.4,  0.5,  0.0], [ 0.0,  0.8,  0.9]])
     ```

   - **Structured Pruning:** This method removes entire filters or channels. For example, if we prune 50% of the filters in `conv1`, the remaining filters might look like this:

     ```python
     tensor([[[0.5, 0.1, 0.2], [0.4, 0.6, 0.3], [0.9, 0.7, 0.8]],
             [[0.5, 0.1, 0.2], [0.4, 0.6, 0.3], [0.9, 0.7, 0.8]]])
     ```

3. **Training:**
   - **Fine-Tuning:** Continue training the pruned model to allow it to adjust to the new, sparser structure. This step is crucial to recover any lost accuracy due to pruning.

### Different Pruning Strategies

- **Global Pruning:** Prune the weights with the smallest magnitude across the entire network. This can lead to better performance than layer-wise pruning because it removes the least important weights regardless of their location.
- **Layer-Wise Pruning:** Prune a fixed percentage of weights in each layer. This approach is simpler but may not be as effective as global pruning.
- **Iterative Pruning:** Prune the model gradually over several training iterations. After each pruning step, fine-tune the model to recover accuracy before pruning again.

### Pruning Code Differences

- Apply pruning:

  ```python
  import torch.nn.utils.prune as prune
  
  # Example: Prune 40% of weights in conv1 layer
  prune.l1_unstructured(model.conv1, name='weight', amount=0.4)
  ```

## 2. Quantization

### Quantization Overview

Quantization reduces the precision of the model's weights and activations, typically from 32-bit floating point to 8-bit integer. This reduces the model size and can improve inference speed.

**Example with BabyCNN:**

- **Model Preparation:** Define the model and set the quantization configuration.
- **Training:** Train the model (for QAT) or simply convert it (for PTQ).
- **Conversion:** Convert the model to a quantized version for inference.

### How Quantization Works

#### Quantization-Aware Training (QAT)

1. **Model Preparation:**
    - Set the QAT configuration:

      ```python
      model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
      ```

    - Example tensor before quantization:

      ```python
      tensor([0.1234, 0.5678, 0.9101])
      ```

    - Example tensor after quantization:

      ```python
      tensor([0.1, 0.6, 0.9], dtype=torch.qint8)
      ```

2. **Training with QAT:**
    - Train the model while simulating quantization effects:

      ```python
      torch.quantization.prepare_qat(model, inplace=True)
      ```

3. **Model Conversion:**
    - Convert the model for quantized inference:

      ```python
      torch.quantization.convert(model, inplace=True)
      ```

#### Post-Training Quantization (PTQ)

1. **Model Preparation:**
    - Set the PTQ configuration:

      ```python
      model.qconfig = torch.quantization.default_qconfig
      ```

2. **Calibration:**
    - Run a subset of data through the model to calibrate for quantization:

      ```python
      with torch.no_grad():
          for data, _ in train_loader:
              model(data)
      ```

3. **Model Conversion:**
    - Convert the model for quantized inference:

      ```python
      torch.quantization.convert(model, inplace=True)
      ```

**Example with BabyCNN and 4 GPUs:**

- **Model Preparation:** Define BabyCNN with quantization configurations.
- **Forward Pass:** Quantize inputs and perform computations with lower precision.
  - Suppose the input tensor before quantization:

    ```python
    tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    ```

  - After quantization:

    ```python
    tensor([[10, 20], [30, 40]], dtype=torch.qint8)
    ```

### Quantization Code Differences

- QAT configuration:

  ```python
  model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
  ```

- PTQ configuration:

  ```python
  model.qconfig = torch.quantization.default_qconfig
  ```

- Prepare for QAT:

  ```python
  torch.quantization.prepare_qat(model, inplace=True)
  ```

- Prepare for PTQ:

  ```python
  torch.quantization.prepare(model, inplace=True)
  ```

- Convert model:

  ```python
  torch.quantization.convert(model, inplace=True)
  ```
