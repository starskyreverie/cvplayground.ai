#!/bin/bash

create_boilerplate() {
  local framework=$1
  local main_file=$2
  local extension=$3
  local train_content=$4
  local infer_content=$5
  local setup_script=$6
  local makefile_content=$7
  local dependencies=$8
  local setup_instructions=$9
  
  mkdir -p $framework
  echo -e "# $framework Implementation of BabyCNN\n\n## Overview\nThis folder contains the implementation of BabyCNN using $framework.\n\n## Setup Instructions\n### Dependencies\n$dependencies\n\n### Setup\n\`\`\`sh\n$setup_instructions\n\`\`\`\n\n## Training\nRun the following command to train the model:\n\`\`\`sh\npython train.$extension\n\`\`\`\n\n## Inference\nRun the following command to perform inference:\n\`\`\`sh\npython infer.$extension\n\`\`\`" > $framework/README.md
  echo -e "$train_content" > $framework/train.$extension
  echo -e "$infer_content" > $framework/infer.$extension

  if [ -n "$setup_script" ]; then
    echo -e "$setup_script" > $framework/setup.sh
    chmod +x $framework/setup.sh
  fi

  if [ -n "$makefile_content" ]; then
    echo -e "$makefile_content" > $framework/Makefile
  fi
}

# PyTorch
create_boilerplate "pytorch" "python" "py" \
  'import torch\n# PyTorch training script\nif __name__ == "__main__":\n    print("Training with PyTorch")' \
  'import torch\n# PyTorch inference script\nif __name__ == "__main__":\n    print("Inference with PyTorch")' \
  '' \
  '' \
  '- torch\n- torchvision' \
  'pip install torch torchvision'

# JAX
create_boilerplate "jax" "python" "py" \
  'import jax\n# JAX training script\nif __name__ == "__main__":\n    print("Training with JAX")' \
  'import jax\n# JAX inference script\nif __name__ == "__main__":\n    print("Inference with JAX")' \
  '' \
  '' \
  '- jax\n- flax' \
  'pip install jax flax'

# TensorFlow 2
create_boilerplate "tf2" "python" "py" \
  'import tensorflow as tf\n# TensorFlow 2 training script\nif __name__ == "__main__":\n    print("Training with TensorFlow 2")' \
  'import tensorflow as tf\n# TensorFlow 2 inference script\nif __name__ == "__main__":\n    print("Inference with TensorFlow 2")' \
  '' \
  '' \
  '- tensorflow' \
  'pip install tensorflow'

# C
create_boilerplate "c" "c" "c" \
  '#include <stdio.h>\n\nint main() {\n    printf("Training with C\\n");\n    return 0;\n}' \
  '#include <stdio.h>\n\nint main() {\n    printf("Inference with C\\n");\n    return 0;\n}' \
  '' \
  'all: train infer\n\ntrain: train.c\n\tgcc -o train train.c\n\ninfer: infer.c\n\tgcc -o infer infer.c\n\nclean:\n\trm -f train infer' \
  '- gcc' \
  'gcc -o train train.c\ngcc -o infer infer.c'

# CUDA
create_boilerplate "cuda" "cuda" "cu" \
  '#include <iostream>\n\nint main() {\n    std::cout << "Training with CUDA" << std::endl;\n    return 0;\n}' \
  '#include <iostream>\n\nint main() {\n    std::cout << "Inference with CUDA" << std::endl;\n    return 0;\n}' \
  '' \
  'all: train infer\n\ntrain: train.cu\n\tnvcc -o train train.cu\n\ninfer: infer.cu\n\tnvcc -o infer infer.cu\n\nclean:\n\trm -f train infer' \
  '- nvcc (CUDA Toolkit)' \
  'nvcc -o train train.cu\nnvcc -o infer infer.cu'

# CUTLASS
create_boilerplate "cutlass" "cutlass" "cu" \
  '#include <iostream>\n\nint main() {\n    std::cout << "Training with CUTLASS" << std::endl;\n    return 0;\n}' \
  '#include <iostream>\n\nint main() {\n    std::cout << "Inference with CUTLASS" << std::endl;\n    return 0;\n}' \
  '' \
  'all: train infer\n\ntrain: train.cu\n\tnvcc -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include -o train train.cu\n\ninfer: infer.cu\n\tnvcc -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include -o infer infer.cu\n\nclean:\n\trm -f train infer' \
  '- CUTLASS\n- nvcc (CUDA Toolkit)' \
  'export CUTLASS_PATH=/path/to/cutlass\nnvcc -I$CUTLASS_PATH/include -I$CUTLASS_PATH/tools/util/include -o train train.cu\nnvcc -I$CUTLASS_PATH/include -I$CUTLASS_PATH/tools/util/include -o infer infer.cu'

# Triton
create_boilerplate "triton" "python" "py" \
  'import triton\n# Triton training script\nif __name__ == "__main__":\n    print("Training with Triton")' \
  'import triton\n# Triton inference script\nif __name__ == "__main__":\n    print("Inference with Triton")' \
  'pip install triton' \
  '' \
  '- triton' \
  'pip install triton'

# MLX (Python)
create_boilerplate "mlx" "python" "py" \
  'import mlx.core as mlx\n# MLX training script\nif __name__ == "__main__":\n    print("Training with MLX")' \
  'import mlx.core as mlx\n# MLX inference script\nif __name__ == "__main__":\n    print("Inference with MLX")' \
  'pip install mlx' \
  '' \
  '- mlx' \
  'pip install mlx'

echo "Boilerplate files and setup scripts created."
