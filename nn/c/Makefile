all: train infer

train: train.c
	gcc -o train train.c

infer: infer.c
	gcc -o infer infer.c

clean:
	rm -f train infer
