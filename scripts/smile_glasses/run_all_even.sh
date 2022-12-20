#!/bin/bash

data_seed=("1111" "2222" "3333" "4444" "5555")
model_seed=("2341" "4578" "3456" "4567" "8910")

for index in ${!data_seed[*]}; do
    for var in 4 6 8 10 12 14 16 18 20
    do
        warmup=100
	pycmd="main.py -t smile -a glasses --pi 1 -ni 5000 -c 0 -nc 3 -nw $var -ds ${data_seed[$index]} -ms ${model_seed[$index]} --project attack11"
	echo $pycmd
	python $pycmd
    done
done

