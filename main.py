import numpy as np
import random
import wandb
import time
import argparse
import os

from distributed_sgd_passive_IFCA import train
from inference_attack_IFCA import evaluate



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed SGD')
    parser.add_argument('--project', help='proejct name', default='attack')
    
    parser.add_argument('-t', help='Main task', default='gender')
    parser.add_argument('-a', help='Target attribute', default='race')
    parser.add_argument('--pi', help='Property id', type=int, default=2)  # black (2)
    parser.add_argument('--pp', help='Property probability', type=float, default=0.5)
    parser.add_argument('-lr', help='Property probability', type=float, default=0.01)
    parser.add_argument('-bs', help='batch size', type=int, default=32)
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
    parser.add_argument('-clip', help='clipping norm',type=float, default=4)
    parser.add_argument('-ep', help='Epsilon for DP', type=float, default=1.0)
    parser.add_argument('-mia', help='mia on', action='store_true', default=False)
    parser.add_argument('-ldp', help='LDP on', action='store_true', default=False)
    parser.add_argument('-cdp', help='CDP on', action='store_true', default=False)
    parser.add_argument('--model_type', help='model type : nm_cnn, rnn, alexnet', default='nm_cnn')
    parser.add_argument('--data_type', help='data type, yelp-author or lfw', default='lfw')
    
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

    wandb.init(project=args.project, entity='sor')
    wandb.config.update(args)

    start_time = time.time()
    filename = train(args.t, args.a, args.pi, args.pp, args.nw, args.nc, args.ni, args.van, args.b, args.k, args.ts, args.c,
                         seed_data, seed_main,args)
    if not args.mia:
        evaluate(filename)
 
    duration = (time.time() - start_time)
    print("SGD ended in %0.2f hour (%.3f sec) " % (duration / float(3600), duration))
