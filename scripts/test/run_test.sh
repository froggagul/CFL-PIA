#!/bin/bash

data_seed=("3333")
model_seed=("3456")

for index in ${!data_seed[*]}; do
    for var in 4
    do
        warmup=100
	pycmd="main.py -t smile -a glasses --pi 1 -ni 150 -c 0 -nc 3 -nw $var -ds ${data_seed[$index]} -ms ${model_seed[$index]} --project test"
	echo $pycmd
	python $pycmd
    done
done

