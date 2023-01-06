#!/bin/bash

data_seed=("3333")
model_seed=("3456")

for index in ${!data_seed[*]}; do
    for var in 5
    do
        warmup=100
	pycmd="main.py -t stars -a user_id --pi 1 -ni 150 -c 1 -nc 3 -nw $var -ds ${data_seed[$index]} -ms ${model_seed[$index]} --project test --data_type yelp-author --model_type yelp_author_gru"
	echo $pycmd
	python $pycmd
    done
done

