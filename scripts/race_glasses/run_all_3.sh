#!/bin/bash

data_seed=("3333" "4444" "5555")
model_seed=("3456" "4567" "8910")

for index in ${!data_seed[*]}; do
    for var in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    do
        warmup=100
	pycmd="main.py -t race -a glasses --pi 1 -ni 6000 -c 1 -nc 3 -nw $var -ds ${data_seed[$index]} -ms ${model_seed[$index]} --project attack10"
	echo $pycmd
	python $pycmd
    done
done

