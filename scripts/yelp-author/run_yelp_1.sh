#!/bin/bash

data_seed=("1111" "2222" "3333" "4444" "5555")
model_seed=("2341" "4578" "3456" "4567" "8910")
gpu=0
for index in ${!data_seed[*]}; do
    for var in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    do
        warmup=100
	pycmd="main.py -t stars -a user_id --pi 1 -ni 150 -lr 0.05 -c ${gpu} -nc 3 -nw $var -ds ${data_seed[$index]} -ms ${model_seed[$index]} --project yelp-author --data_type yelp-author --model_type yelp_author_gru"
	echo $pycmd
	python $pycmd
    done
done

