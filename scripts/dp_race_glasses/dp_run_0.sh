#!/bin/bash

data_seed=("5555")
model_seed=("8910")
eps="0.5"
gpu="0"
for index in ${!data_seed[*]}; do
    for var in 5 10 15 20
    do
        warmup=100
	pycmd="main.py -t race -a glasses --pi 1 -ni 5000 -c ${gpu} -nc 3 -nw $var -ds ${data_seed[$index]} -ms ${model_seed[$index]} --project attack_dp -dp -ep ${eps}"
	echo $pycmd
	python $pycmd
    done
done

