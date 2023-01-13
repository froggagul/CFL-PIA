#!/bin/bash

task=("smile")
attack=("glasses")

for index in ${!task[*]}; do
    for nc in 4 5 6
    do
	pycmd="main.py -t ${task[$index]} -a ${attack[$index]} --pi 1 -ni 5000 -c 1 -nc $nc -nw 20 -ds 1111 -ms 1234 --project cluster_number"
	echo $pycmd
	python $pycmd
    done
done

