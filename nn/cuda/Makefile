all: train infer

train: train.cu
	nvcc -o train train.cu

infer: infer.cu
	nvcc -o infer infer.cu

clean:
	rm -f train infer
