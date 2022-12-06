import numpy as np
import random
import wandb
import time
import argparse
import os

from distributed_sgd_passive_IFCA import train_lfw
from inference_attack_IFCA import evaluate_lfw



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed SGD')
    parser.add_argument('--project', help='proejct name', default='attack')
    
    parser.add_argument('-t', help='Main task', default='gender')
    parser.add_argument('-a', help='Target attribute', default='race')
    parser.add_argument('--pi', help='Property id', type=int, default=2)  # black (2)
    parser.add_argument('--pp', help='Property probability', type=float, default=0.5)
    parser.add_argument('-nw', help='# of workers', type=int, default=30)
    parser.add_argument('-nc', help='# of clusters', type=int, default=3)
    parser.add_argument('-ni', help='# of iterations', type=int, default=6000)
    parser.add_argument('--van', help='victim_all_nonproperty', action='store_true')
    parser.add_argument('--b', help='balance', action='store_true')
    parser.add_argument('-k', help='k', type=int, default=5)
    parser.add_argument('-ds', help='data seed (-1 for time-dependent seed)', type=int, default=54321)
    parser.add_argument('-ms', help='main seed (-1 for time-dependent seed)', type=int, default=12345)
    parser.add_argument('--ts', help='Train size', type=float, default=0.3)
    parser.add_argument('-c', help='CUDA num (-1 for CPU-only)', default=-1)

    args = parser.parse_args()

    if args.ds == -1:
        seed_data = time.time()
    else:
        seed_data = args.ds

    if args.ms == -1:
        seed_main = time.time() + 1234
    else:
        seed_main = args.ms

    args.ms = seed_main
    args.ds = seed_data

    wandb.init(project=args.project, entity='froggagul')
    wandb.config.update(args)

    start_time = time.time()
    filename = train_lfw(args.t, args.a, args.pi, args.pp, args.nw, args.nc, args.ni, args.van, args.b, args.k, args.ts, args.c,
              seed_data, seed_main)
    evaluate_lfw(filename)
 
    duration = (time.time() - start_time)
    print("SGD ended in %0.2f hour (%.3f sec) " % (duration / float(3600), duration))
