all: train infer

train: train.cu
	nvcc -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include -o train train.cu

infer: infer.cu
	nvcc -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include -o infer infer.cu

clean:
	rm -f train infer
