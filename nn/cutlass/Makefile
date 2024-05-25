all: train

train: train.cu
	nvcc -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include -o train train.cu

clean:
	rm -f train
