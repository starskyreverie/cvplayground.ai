all: train

train: train.cu
	nvcc -o train train.cu

clean:
	rm -f train
