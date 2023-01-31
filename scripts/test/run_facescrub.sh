#!/bin/bash

data_seed=("3333")
model_seed=("3456")

for index in ${!data_seed[*]}; do
    for var in 5
    do
        warmup=100
	pycmd="main.py -t gender -a name --pi 0 -ni 150 -c 1 -nc 3 -nw $var -ds ${data_seed[$index]} -ms ${model_seed[$index]} --project test --data_type face_scrub --model_type nm_cnn"
	echo $pycmd
	python $pycmd
    done
done

