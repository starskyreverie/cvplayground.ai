#!/bin/bash

NUM_GPUS=4  # Change this to the number of GPUs you have

python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train_pippy.py
