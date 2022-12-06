#!/bin/bash

for var in {3..20}
do 
    echo $var
    echo ./distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw $var
    /home/hjkim/anaconda3/envs/fl/bin/python ./distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw $var
    echo ./inference_attack_IFCA.py -t gender -a race --pi 2 -nw $var
    /home/hjkim/anaconda3/envs/fl/bin/python ./inference_attack_IFCA.py -t gender -a race --pi 2 -nw $var
done
