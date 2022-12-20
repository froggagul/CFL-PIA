#!/bin/bash

data_seed=("54322")
model_seed=("22345")

for index in ${!data_seed[*]}; do
    for var in 3 4 5
    do
        warmup=100
	pycmd="main.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -nw $var -ds ${data_seed[$index]} -ms ${model_seed[$index]} --project tsne"
	echo $pycmd
	python $pycmd
    done
done

