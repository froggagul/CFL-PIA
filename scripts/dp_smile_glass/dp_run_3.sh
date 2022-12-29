#!/bin/bash

data_seed=("5555")
model_seed=("8910")
eps="0.05"
gpu="1"
for index in ${!data_seed[*]}; do
    for var in 5 10 15 20 25
    do
        warmup=100
	pycmd="main.py -t smile -a glasses --pi 1 -ni 5000 -c ${gpu} -nc 3 -nw $var -ds ${data_seed[$index]} -ms ${model_seed[$index]} --project attack9 -dp -ep ${eps}"
	echo $pycmd
	python $pycmd
    done
done

