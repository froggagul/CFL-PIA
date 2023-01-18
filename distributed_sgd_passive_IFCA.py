import argparse
import copy
import logging
import os
import pickle
import random
import time
import warnings
from datetime import datetime
from sklearn.metrics import f1_score

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.simplefilter(action="ignore", category=FutureWarning)

from collections import OrderedDict

from metrics import AUROC as AUROCMetric
import wandb

from torchdp.privacy_engine import PrivacyEngine
from torchdp.per_sample_gradient_clip import PerSampleGradientClipper
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

from models import Model, ModelFactory
from constant import GRAD_SAVE_DIR, MODEL_SAVE_DIR
from checkpoint import ExperimentCheckpoint
from data import load_data, load_attr, prepare_data_biased,prepare_mia_data_biased

if not os.path.exists(GRAD_SAVE_DIR):
    os.mkdir(GRAD_SAVE_DIR)

# seed가 항상 동일하게 작용하도록 (재현성)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 로깅 설정
# 폴더 생성
os.makedirs("./log_1", exist_ok=True)

# logger instance 생성
logger = logging.getLogger(__name__)

# formatter 생성
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# handler 생성 (stream, file)
streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler("./log_1/" + datetime.now().strftime('log_fl_%Y_%m_%d_%H_%M.log'))

# logger instance에 fomatter 설정
streamHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# logger instance에 handler 설정
logger.addHandler(streamHandler)
logger.addHandler(fileHandler)

# logger instnace로 log 찍기
logger.setLevel(level=logging.INFO)


def np_to_one_hot(targets, classes):  # 정수 numpy to one-hot encoding tensor
    targets_tensor = torch.from_numpy(targets.astype(np.int64))
    targets = torch.zeros(targets.shape[0], classes).scatter_(1, targets_tensor.unsqueeze(1), 1.0)
    return targets


def weights_init(m):  # weight 초기화
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape).to(device)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, targets_B=None):  # batch를 가져옴
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if targets_B is None:
            yield inputs[excerpt], targets[excerpt]
        else:
            yield inputs[excerpt], targets[excerpt], targets_B[excerpt]

def gen_batch(x, y, args,n=1):
    for i, v in enumerate(x):
        if args.mia:
            y_slice = y[i]
        else:
            y_slice = y[i][:, 0]
        l = len(v)
        for ndx in range(0, l, n):
            yield v[ndx:min(ndx + n, l)], y_slice[ndx:min(ndx + n, l)]


def train(task='gender', attr='race', prop_id=2, p_prop=0.5, n_workers=2, n_clusters=3, num_iteration=3000,
              victim_all_nonprop=False, balance=False, k=5, train_size=0.3, cuda=-1, seed_data=54321, seed_main=12345,args=None):
    
    x, y, prop = load_data(args.data_type, task, attr)
    if not args.mia:
        BINARY_ATTRS, MULTI_ATTRS = load_attr(args.data_type)
        prop_dict = MULTI_ATTRS[attr] if attr in MULTI_ATTRS else BINARY_ATTRS[attr]

        logger.info('Training {} and infering {} property {} with {} data'.format(task, attr, prop_dict[prop_id], len(x)))

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    #prop = np.asarray(prop, dtype=np.int32)  # property label인지 (1) 아닌지 (0)
    prop = np.where(np.asarray(prop, dtype=np.int32) == prop_id, 1, 0)


    filename = wandb.run.name

    print(x.shape, y.shape, prop.shape)
    if not args.mia:
        filename = f"{args.project}/{args.t}_{args.a}_{args.pi}_{args.nw}_{wandb.run.name}_{args.ds}"
        if args.ldp:
            filename = f"{args.project}/ldp_{args.t}_{args.a}_{args.pi}_{args.nw}_{args.ep}_{args.clip}_{wandb.run.name}_{args.ds}"
        elif args.cdp:
            filename = f"{args.project}/cdp_{args.t}_{args.a}_{args.pi}_{args.nw}_{args.ep}_{args.clip}_{wandb.run.name}_{args.ds}"
    else:
        filname =  f"{args.project}/mia_{args.data_type}_{args.nw}_{args.nc}_{wandb.run.name}_{args.ds}"


    # indices = np.arange(len(x))
    # prop_indices = indices[prop == prop_id]
    # nonprop_indices = indices[prop != prop_id]

    # prop[prop_indices] = 1
    # prop[nonprop_indices] = 0

    # filename = "lfw_psMT_{}_{}_{}_alpha{}_k{}_nc{}".format(task, attr, prop_id, 0, k, n_clusters)

    # if n_workers > 2:
    #     filename += '_n{}'.format(n_workers)

    if args.mia:
        train_multi_task_ps(
            (x, y),
            input_shape=(3, 62, 47),
            p_prop=p_prop,  # balance=balance,
            filename=filename,
            n_workers=n_workers,
            n_clusters=n_clusters,
            lr = args.lr,
            k=k,
            num_iteration=num_iteration,
            victim_all_nonprop=victim_all_nonprop,
            train_size=train_size,
            cuda=cuda,
            seed_data=seed_data,
            seed_main=seed_main,
            args= args
        )
    else:
        train_multi_task_ps(
            (x, y, prop),
            input_shape=(3, 62, 47),
            p_prop=p_prop,  # balance=balance,
            filename=filename,
            n_workers=n_workers,
            n_clusters=n_clusters,
            lr = args.lr,
            k=k,
            num_iteration=num_iteration,
            victim_all_nonprop=victim_all_nonprop,
            train_size=train_size,
            cuda=cuda,
            seed_data=seed_data,
            seed_main=seed_main,
            args= args
        )
    return filename

def build_worker(model_type, input_shape, classes=2, lr=None, device='cpu', args=None):  # worker 1개 생성하고 초기화
    kwargs = {
            "classes": classes,
            "input_shape": input_shape,
            "lr": lr,
            "args": args
        }

    worker: Model = ModelFactory.create(
        model_type,
        **kwargs
    )
    worker = worker.to(device)
    worker.apply(weights_init)
    worker.train()

    return worker


def inf_data(x, y, batchsize, shuffle=False, y_b=None):  # random batch 무한히 가져오기
    while True:
        for b in iterate_minibatches(x, y, batchsize=batchsize, shuffle=shuffle, targets_B=y_b):
            yield b


def mix_inf_data(p_inputs, p_targets, np_inputs, np_targets, batchsize,
                 mix_p=0.5):  # prop - nonprop을 섞음 (train용 gradient 얻을 때 사용됨)
    p_batchsize = int(mix_p * batchsize)
    np_batchsize = batchsize - p_batchsize

    logger.info('Mixing {} prop data with {} non prop data'.format(p_batchsize, np_batchsize))

    p_gen = inf_data(p_inputs, p_targets, p_batchsize, shuffle=True)
    np_gen = inf_data(np_inputs, np_targets, np_batchsize, shuffle=True)

    while True:
        px, py = next(p_gen)
        npx, npy = next(np_gen)
        x = np.vstack([px, npx])
        y = np.concatenate([py, npy])
        yield x, y


def set_local(global_params, local_params):  # global model을 모든 worker에게 다 적용
    with torch.no_grad():
        for device in local_params:
            for param in list(device.keys()):
                if param in global_params:
                    device[param].data.copy_(global_params[param].data)


def set_local_single(global_params, local_param):  # global model을 하나의 worker에게 적용
    with torch.no_grad():
        for param in list(local_param.keys()):
            if param in global_params:
                local_param[param].data.copy_(global_params[param].data)


def update_global(global_params, grads_dict, lr, num_data):  # worker가 학습한 gradient를 반영해서 global model을 update
    with torch.no_grad():
        for key in list(global_params.keys()):
            if key in grads_dict:
                glob_param = global_params[key]
                local_grad = grads_dict[key]
                glob_param.data.copy_(glob_param.data - local_grad.data * lr / num_data)


def add_nonprop(test_prop_indices, nonprop_indices, p_prop=0.7):
    n = len(test_prop_indices)
    n_to_add = int(n / p_prop) - n

    sampled_non_prop = np.random.choice(nonprop_indices, n_to_add, replace=False)
    nonprop_indices = np.setdiff1d(nonprop_indices, sampled_non_prop)
    return sampled_non_prop, nonprop_indices

def print_index(index_list, count):
    count[0] += 1
    print_string = str(count[0]) + ': '

    for i, v in enumerate(index_list):
        if i == len(index_list) - 1:
            print_string = print_string + str(v)
        else:
            print_string = print_string + str(v) + ' | '

    logger.info(print_string)

'''def gradient_getter(data, p_g, p_indices, fn, batch_size=32, shuffle=True):
    X, y = data
    p_x, p_y = X[p_indices], y[p_indices]

    for batch in iterate_minibatches(p_x, p_y, batch_size, shuffle=shuffle):
        xx, yy = batch
        gs = fn(xx, yy)
        p_g.append(np.asarray(gs).flatten())


def gradient_getter_with_gen(data_gen, p_g, fn, iters=10, param_names=None):
    for _ in range(iters):
        xx, yy = next(data_gen)
        gs = fn(xx, yy)
        if isinstance(gs, dict):
            gs = collect_grads(gs, param_names)
        else:
            gs = np.asarray(gs).flatten()
        p_g.append(gs)'''


def gradient_getter_with_gen_multi(data_gen1, data_gen2, p_g, fn, device='cpu', iters=10,
                                   n_workers=5):  # train용 gradient를 생성하기 위해 FL을 emulate함 (cluster FL이 적용되도록 추가 수정 필요)
    for _ in range(iters):
        xx, yy = next(data_gen1)
        fn.optimizer.zero_grad()
        presult = fn(torch.from_numpy(xx).to(device)).cpu()
        ptargets = torch.from_numpy(yy).to(dtype=torch.long)
        loss = fn.criterion(presult, ptargets)
        loss.backward()
        pgs = {}
        for name, param in fn.named_parameters():
            if param.requires_grad:
                pgs[name] = param.grad.cpu().data

        if isinstance(pgs, dict):
            for key in pgs:
                pgs[key] = np.asarray(pgs[key])
        else:
            pgs = np.asarray(pgs).flatten()

        for _ in range(n_workers - 2):
            xx, yy = next(data_gen2)
            fn.optimizer.zero_grad()
            npresult = fn(torch.from_numpy(xx).to(device)).cpu()
            nptargets = torch.from_numpy(yy).to(dtype=torch.long)
            loss = fn.criterion(npresult, nptargets)
            loss.backward()
            npgs = {}
            for name, param in fn.named_parameters():
                if param.requires_grad:
                    npgs[name] = param.grad.cpu().data

            if isinstance(npgs, dict):
                for key in npgs:
                    pgs[key] += np.asarray(npgs[key])
            else:
                npgs = np.asarray(npgs).flatten()
                pgs += npgs

        if isinstance(pgs, dict):
            pgs = collect_grads(pgs)

        p_g.append(pgs)


def collect_grads(grads_dict, avg_pool=False,
                  pool_thresh=5000):  # convolution gradient를 하나의 vector로 만듬. 너무 크면 pooling을 적용해서 크기를 줄인다.
    g = []
    for param_name in grads_dict:
        grad = grads_dict[param_name]
        # grad = np.asarray(grad)
        shape = grad.shape

        if len(shape) == 1:
            continue

        grad = np.abs(grad)
        if len(shape) == 4:
            if shape[0] * shape[1] > pool_thresh:
                continue
            grad = grad.reshape(shape[0], shape[1], -1)

        if len(shape) > 2 or shape[0] * shape[1] > pool_thresh:
            if avg_pool:
                grad = np.mean(grad, -1)
            else:
                grad = np.max(grad, -1)

        g.append(grad.flatten())

    g = np.concatenate(g)
    return g


def aggregate_dicts(dicts):  # attacker를 제외한 모델의 gradient를 더함 (공격자 입장에서는 두 iteration global model weight들의 차이를 구한 것)
    aggr_dict = dicts[0]

    for key in aggr_dict:
        aggr_dict[key] = np.asarray(aggr_dict[key].cpu().data)

    for d in dicts[1:]:
        for key in aggr_dict:
            aggr_dict[key] += np.asarray(d[key].cpu().data)

    return collect_grads(aggr_dict)


# active property inference
def train_multi_task_ps(data, num_iteration=6000, train_size=0.3, victim_id=0, warm_up_iters=100,
                        input_shape=(None, 3, 50, 50), n_workers=2, n_clusters=3, lr=0.01, attacker_id=1,
                        filename="data",
                        p_prop=0.5, victim_all_nonprop=True, k=5, cuda=-1, seed_data=54321, seed_main=12345,args=None):
    if cuda == '-1':
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda:' + cuda)
    else:
        device = torch.device('cpu')

    if args.mia:
        splitted_X, splitted_y, X_test, y_test, splitted_X_test, splitted_y_test = prepare_mia_data_biased(data, train_size,
                                                                                                       n_workers,
                                                                                                       seed=seed_data,
                                                                                                       # non-iid dataset 생성 --> worker별로 데이터셋이 할당됨
                                                                                                       victim_all_nonprop=victim_all_nonprop,
                                                                                                       p_prop=p_prop,args=args)
    else:
        splitted_X, splitted_y, X_test, y_test, splitted_X_test, splitted_y_test = prepare_data_biased(data, train_size,
                                                                                                       n_workers,
                                                                                                       seed=seed_data,
                                                                                                       # non-iid dataset 생성 --> worker별로 데이터셋이 할당됨
                                                                                                       victim_all_nonprop=victim_all_nonprop,
                                                                                                       p_prop=p_prop,args=args)

    torch.manual_seed(seed_main)
    torch.cuda.manual_seed(seed_main)
    torch.cuda.manual_seed_all(seed_main)  # if use multi-GPU
    np.random.seed(seed_main)
    random.seed(seed_main)

    if args.mia:
        y_test = y_test
    else:
        p_test = y_test[:, 1]
        y_test = y_test[:, 0]

    classes = len(np.unique(y_test))
    # build test network
    args_dp = copy.deepcopy(args)
    args_dp.dp = False
    model_type = args.model_type

    network_global = build_worker(model_type, classes=classes, input_shape=input_shape, args=args_dp, device=device, lr=lr)  # global model
    global_params = network_global.get_params()

    # build clusters
    cluster_networks = []
    cluster_params = []
    for i in range(n_clusters):
        network = build_worker(model_type, classes=classes, input_shape=input_shape, args=args_dp, device=device, lr=lr)
        params = network.get_params()

        cluster_networks.append(network)
        cluster_params.append(params)

    # build local workers
    worker_networks = []
    worker_params = []
    data_gens = []

    worker_networks_IFCA = []
    worker_params_IFCA = []

    for i in range(n_workers):  # worker 생성
        # generator 생성
        if i == attacker_id and not args.mia:  # attacker
            split_y = splitted_y[i]

            data_gen = inf_data(splitted_X[i], split_y[:, 0], y_b=split_y[:, 1], batchsize=args.bs, shuffle=True)
            data_gens.append(data_gen)

            logger.info('Participant {} with {} data'.format(i, len(splitted_X[i])))
        elif i == victim_id and not args.mia:  # victim
            vic_X = np.vstack([splitted_X[i][0], splitted_X[i][1]])
            vic_y = np.concatenate([splitted_y[i][0][:, 0], splitted_y[i][1][:, 0]])
            vic_p = np.concatenate([splitted_y[i][0][:, 1], splitted_y[i][1][:, 1]])

            data_gen = inf_data(vic_X, vic_y, y_b=vic_p, batchsize=args.bs, shuffle=True)
            data_gen_p = inf_data(splitted_X[i][0], splitted_y[i][0][:, 0], batchsize=args.bs, shuffle=True)
            data_gen_np = inf_data(splitted_X[i][1], splitted_y[i][1][:, 0], batchsize=args.bs, shuffle=True)

            data_gens.append(data_gen)
            logger.info('Participant {} with {} data'.format(i, len(splitted_X[i][0]) + len(splitted_X[i][1])))
        else:

            if args.mia:
                data_gen = inf_data(splitted_X[i], splitted_y[i], batchsize=args.bs, shuffle=True)
            else:
                data_gen = inf_data(splitted_X[i], splitted_y[i][:, 0], batchsize=args.bs, shuffle=True)
            data_gens.append(data_gen)

            logger.info('Participant {} with {} data'.format(i, len(splitted_X[i])))

        network = build_worker(model_type, input_shape, classes=classes, lr=lr, device=device,args=args)  # worker들 생성
        worker_networks.append(network)
        params = network.get_params()
        worker_params.append(params)

        network_IFCA = build_worker(model_type, input_shape, classes=classes, lr=lr, device=device,args=args)  # CFL용 worker들 생성
        worker_networks_IFCA.append(network_IFCA)
        params = network_IFCA.get_params()
        worker_params_IFCA.append(params)

    # container for gradients
    train_pg, train_npg = [], []
    test_pg, test_npg = [], []
    train_cluster_nv_pg, train_cluster_nv_npg = [], []
    test_cluster_nv_pg, test_cluster_nv_npg = [], []
    for j in range(n_clusters):
        train_cluster_nv_pg.append([])
        train_cluster_nv_npg.append([])
        test_cluster_nv_pg.append([])
        test_cluster_nv_npg.append([])

    if args.mia:
        X, y = data
    else:
        X, y, _ = data


    if not args.mia:
        # attacker's aux data
        X_adv, y_adv = splitted_X[attacker_id], splitted_y[attacker_id]
        p_adv = y_adv[:, 1]
        y_adv = y_adv[:, 0]

        indices = np.arange(len(X_adv))
        prop_indices = indices[p_adv == 1]
        nonprop_indices = indices[p_adv == 0]
        adv_gen = mix_inf_data(X_adv[prop_indices], splitted_y[attacker_id][prop_indices],                            X_adv[nonprop_indices], splitted_y[attacker_id][nonprop_indices], batchsize=args.bs,                            mix_p=0.2)  # 공격자용 data generator

        X_adv = np.vstack([X_adv, X_test])
        y_adv = np.concatenate([y_adv, y_test])
        p_adv = np.concatenate([p_adv, p_test])

        indices = np.arange(len(p_adv))
        train_prop_indices = indices[p_adv == 1]
        train_prop_gen = inf_data(X_adv[train_prop_indices], y_adv[train_prop_indices], args.bs, shuffle=True)

        indices = np.arange(len(p_test))
        nonprop_indices = indices[p_test == 0]
        n_nonprop = len(nonprop_indices)

        logger.info('Attacker prop data {}, non prop data {}'.format(len(train_prop_indices), n_nonprop))
        train_nonprop_gen = inf_data(X_test[nonprop_indices], y_test[nonprop_indices], args.bs, shuffle=True)

        train_mix_gens = []  # 학습용 aggregated gradient를 생성할때 다양한 property distribution을 가진 상황을 가정하여 만든 data generator
        for train_mix_p in [0.4, 0.6, 0.8]:
            train_mix_gen = mix_inf_data(X_adv[train_prop_indices], y_adv[train_prop_indices],
                                     X_test[nonprop_indices], y_test[nonprop_indices], batchsize=args.bs, mix_p=train_mix_p)
            train_mix_gens.append(train_mix_gen)

    start_time = time.time()
    for it in range(num_iteration):  # stages 시작
        logger.info("Cur iteration: %d", it)

        aggr_grad = []
        aggr_grad_cluster = []
        for i in range(n_clusters):
            aggr_grad_cluster.append([])
        cluster_global_grads = []
        cluster_global_index = []
        cluster_global_isize = []

        set_local(global_params, worker_params)  # set global model to all devices
        for i in range(n_workers):
            network = worker_networks[i]
            params = worker_params[i]
            data_gen = data_gens[i]

            network.optimizer.zero_grad()
            if i == attacker_id and not args.mia:
                batch = next(adv_gen)
                inputs, targets = batch
                targets = targets[:, 0]
            elif i == victim_id and not args.mia:  # k번째마다 property가 포함됨, 나머지는 포함 X
                if it % k == 0:
                    inputs, targets = next(data_gen_p)
                else:
                    inputs, targets = next(data_gen_np)
            else:
                inputs, targets = next(data_gen)

            input_tensor = torch.from_numpy(inputs).to(device)
            pred = network(input_tensor).cpu()
            targets = torch.from_numpy(targets).to(dtype=torch.long)
            loss = network.criterion(pred, targets)
            loss.backward()

            grads_dict = OrderedDict()

            if args.ldp:
                network.clipper.step()
                for param in params.keys():

                    params[param].grad += gaussian_noise(params[param].shape, args.clip/args.bs, args.ep, device=device)
                    grads_dict[param] = copy.deepcopy(params[param].grad)              
            else:
                if args.cdp:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), args.clip)    
                for param in params.keys():
                    grads_dict[param] = copy.deepcopy(params[param].grad)

            if i != attacker_id:
                aggr_grad.append(grads_dict)  # 공격자를 제외한 gradient 수집
            else:
                wandb.log({
                    "grad/train/loss/attacker/fl": loss.item()
                }, it)

            update_global(global_params, grads_dict, lr, 1.0)  # update

            # IFCA
            network_IFCA = worker_networks_IFCA[i]
            params_IFCA = worker_params_IFCA[i]
            loss_list = []
            grads_list = []
            for j in range(n_clusters):
                # check ith cluster
                #cluster_network = cluster_networks[j]
                cluster_param = cluster_params[j]

                set_local_single(cluster_param, params_IFCA)
                network_IFCA.optimizer.zero_grad()

                pred = network_IFCA(input_tensor).cpu()
                loss = network_IFCA.criterion(pred, targets)
                loss.backward()
                loss_list.append(loss.item())

                grads_dict = OrderedDict()
                if args.ldp:
                    network_IFCA.clipper.step()
                    for param in params.keys():

                        params_IFCA[param].grad += gaussian_noise(params_IFCA[param].shape, args.clip/args.bs, args.ep, device=device)

                        grads_dict[param] = copy.deepcopy(params_IFCA[param].grad)
                else:
                    if args.cdp:
                        torch.nn.utils.clip_grad_norm_(network_IFCA.parameters(), args.clip)
                        
                    for param in params_IFCA.keys():
                        grads_dict[param] = copy.deepcopy(params_IFCA[param].grad)
        
                grads_list.append(grads_dict)

            min_loss = min(loss_list)
            min_index = loss_list.index(min_loss)  # 가장 loss가 낮은 모델에 대해서 update
            # logger.info("Index: %d", min_index)

            if i != attacker_id:
                aggr_grad_cluster[min_index].append(grads_list[min_index])
            else:
                wandb.log({
                    "grad/train/loss/attacker/ifca": loss.item(),
                }, it)
            if i == victim_id:  # victim이 속한 cluster index
                cur_index = min_index

            cluster_global_grads.append(grads_list[min_index])
            cluster_global_index.append(min_index)
            cluster_global_isize.append(inputs.shape[0])

        for i in range(n_workers):  # update clustered global models
            w_index = cluster_global_index[i]
            update_global(cluster_params[w_index], cluster_global_grads[i], lr * args.bs, cluster_global_isize[i])


        if args.cdp:
            for global_param in global_params.keys():
                global_params[global_param].data.add_(gaussian_noise(global_params[global_param].data.shape, args.clip/args.nw, args.ep, device=device))
                
            for i in range(n_clusters):
                for cluster_param in cluster_params[i].keys():
                    cluster_params[i][cluster_param].data.add_(gaussian_noise(cluster_params[i][cluster_param].shape, args.clip/args.nw, args.ep, device=device))
                                            
                
            
        result_count = [0]
        print_index(cluster_global_index, result_count)

        warm_up_iters = 100
        if it >= warm_up_iters and not args.mia:
            
            test_gs = aggregate_dicts(aggr_grad)
            if it % k == 0:  # victim이 property를 가질 때 / 안 가질때 aggregated gradient를 수집
                test_pg.append(test_gs)
            else:
                test_npg.append(test_gs)


            for j in range(n_clusters):
                if len(aggr_grad_cluster[j]) != 0:
                    test_gs = aggregate_dicts(aggr_grad_cluster[j])
                    if it % k == 0 and j == cur_index:
                        test_cluster_nv_pg[j].append(test_gs) # when we don't know victim's cluster
                    else:
                        test_cluster_nv_npg[j].append(test_gs) # when we don't know victim's cluster

            if n_workers > 2:  # 학습용 aggregated gradient 생성
                for train_mix_gen in train_mix_gens:
                    gradient_getter_with_gen_multi(train_mix_gen, train_nonprop_gen, train_pg, network_global,
                                                   device=device,
                                                   iters=2, n_workers=n_workers)
                gradient_getter_with_gen_multi(train_prop_gen, train_nonprop_gen, train_pg, network_global,
                                               device=device,
                                               iters=2, n_workers=n_workers)
                gradient_getter_with_gen_multi(train_nonprop_gen, train_nonprop_gen, train_npg, network_global,
                                               device=device,
                                               iters=8, n_workers=n_workers)

                # when we don't know victim's cluster
                for j in range(n_clusters):

                    for train_mix_gen in train_mix_gens:
                        gradient_getter_with_gen_multi(train_mix_gen, train_nonprop_gen, train_cluster_nv_pg[j],
                                                       cluster_networks[j],
                                                       device=device,
                                                       iters=2, n_workers=int(n_workers/n_clusters))

                    gradient_getter_with_gen_multi(train_prop_gen, train_nonprop_gen, train_cluster_nv_pg[j],
                                                   cluster_networks[j],
                                                   device=device,
                                                   iters=2, n_workers=int(n_workers/n_clusters))

                    gradient_getter_with_gen_multi(train_nonprop_gen, train_nonprop_gen, train_cluster_nv_npg[j],
                                                   cluster_networks[j],
                                                   device=device,
                                                   iters=8, n_workers=int(n_workers/n_clusters))

            '''else: # we only use multi devices
                gradient_getter_with_gen(train_prop_gen, train_pg, global_grad_fn, iters=2,
                                         param_names=params_names)
                for train_mix_gen in train_mix_gens:
                    gradient_getter_with_gen(train_mix_gen, train_pg, global_grad_fn, iters=2,
                                             param_names=params_names)

                gradient_getter_with_gen(train_nonprop_gen, train_npg, global_grad_fn, iters=8,
                                         param_names=params_names)'''

        if (it + 1) % 10 == 0 and it > 0:  # validation

            network_global.eval()
            for j in range(n_clusters):
                cluster_networks[j].eval()

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0

            val_IFCA_acc = 0

            y_true = np.array([], dtype=int)
            y_pred = np.array([], dtype=int)
            y_pred_CFL = np.array([], dtype=int)

            cluster_global_index = []
            ifca_auc_global = AUROCMetric()
            ifca_auc_cluster = [AUROCMetric() for _ in range(n_clusters)]
            
            fl_auc = AUROCMetric()

            with torch.no_grad():
                for batch in gen_batch(splitted_X_test, splitted_y_test, args,args.bs):
                    inputs, targets = batch
                    input_tensor = torch.from_numpy(inputs).to(device)
                    pred = network_global(input_tensor).cpu()
                    targets2 = torch.from_numpy(targets).to(dtype=torch.long)

                    fl_auc.update(targets2, pred)

                    err = network_global.criterion(pred, targets2)
                    y = torch.from_numpy(targets)
                    y_max_scores, y_max_idx = pred.max(dim=1)
                    acc = (y == y_max_idx).sum() / y.size(0)

                    val_err += err.item()
                    val_acc += acc
                    val_batches += 1

                    y_pack = y.cpu().detach().numpy()
                    pred_pack = y_max_idx.cpu().detach().numpy()
                    y_true = np.append(y_true, y_pack)
                    y_pred = np.append(y_pred, pred_pack)

                    loss_list = []
                    pred_list = []
                    for j in range(n_clusters):
                        # check ith cluster
                        cluster_network = cluster_networks[j]
                        
                        pred = cluster_network(input_tensor).cpu()
                        loss = cluster_network.criterion(pred, targets2)
                        loss_list.append(loss.item())
                        pred_list.append(pred)
                        
                        ifca_auc_cluster[j].update(targets2, pred)

                    min_loss = min(loss_list)
                    min_index = loss_list.index(min_loss)
                    cluster_global_index.append(min_index)
                    # logger.info("Val Index: %d", min_index)
                    pred = pred_list[min_index]
                    ifca_auc_global.update(targets2, pred)

                    y_max_scores, y_max_idx = pred.max(dim=1)
                    acc = (y == y_max_idx).sum() / y.size(0)
                    val_IFCA_acc += acc

                    pred_CFL_pack = y_max_idx.cpu().detach().numpy()
                    y_pred_CFL = np.append(y_pred_CFL, pred_CFL_pack)

            print_index(cluster_global_index, result_count)

            logger.info("  Iteration {} of {} took {:.3f}s\n".format(it + 1, num_iteration,
                                                                   time.time() - start_time))
            
            cluster_metrics = {
                f"grad/test/ifca/auc_cluster_{i}": ifca_auc_cluster[i].compute() for i in range(n_clusters)
            }
            wandb.log({
                "grad/test/fl/accuracy": val_acc.item() / val_batches * 100,
                "grad/test/fl/f1": f1_score(y_true, y_pred, average='macro'),
                "grad/test/fl/auc": fl_auc.compute(),
                "grad/test/ifca/accuracy": val_IFCA_acc.item() / val_batches * 100,
                "grad/test/ifca/f1": f1_score(y_true, y_pred_CFL, average='macro'),
                "grad/test/ifca/auc_global": ifca_auc_global.compute(),
                **cluster_metrics
            }, it)
            
            logger.info("  Orig test accuracy: {:.3f}".format(val_acc.item() / val_batches * 100) + " (F1 score: {:.3f}".format(f1_score(y_true, y_pred, average='macro')) + ")")
            logger.info("  IFCA test accuracy: {:.3f}".format(val_IFCA_acc.item() / val_batches * 100) + " (F1 score: {:.3f}".format(f1_score(y_true, y_pred_CFL, average='macro')) + ")\n")

            network_global.train()
            for j in range(n_clusters):
                cluster_networks[j].train()

            start_time = time.time()

    checkpoint = ExperimentCheckpoint(
        network_global,
        cluster_networks,
        worker_networks,
        worker_networks_IFCA,
        args
        )
    try:
        checkpoint.save(os.path.join(MODEL_SAVE_DIR, args.project, filename))
    except:
        print("checkpoint save failed")
    filepath = os.path.join(GRAD_SAVE_DIR, "{}.npz".format(filename))
    os.makedirs(os.path.join(GRAD_SAVE_DIR, args.project), exist_ok=True)
    print(filepath)

    np.savez(filepath,
             train_pg=train_pg, train_npg=train_npg, test_pg=test_pg, test_npg=test_npg,
             train_cluster_nv_pg=train_cluster_nv_pg, train_cluster_nv_npg=train_cluster_nv_npg,
             test_cluster_nv_pg=test_cluster_nv_pg, test_cluster_nv_npg=test_cluster_nv_npg)


if __name__ == '__main__':
    """
    below script is deprecated, see main.py
    """
    # parser = argparse.ArgumentParser(description='Distributed SGD')
    # parser.add_argument('-t', help='Main task', default='gender')
    # parser.add_argument('-a', help='Target attribute', default='race')
    # parser.add_argument('--pi', help='Property id', type=int, default=2)  # black (2)
    # parser.add_argument('--pp', help='Property probability', type=float, default=0.5)
    # parser.add_argument('-nw', help='# of workers', type=int, default=30)
    # parser.add_argument('-nc', help='# of clusters', type=int, default=3)
    # parser.add_argument('-ni', help='# of iterations', type=int, default=5000)
    # parser.add_argument('--van', help='victim_all_nonproperty', action='store_true')
    # parser.add_argument('--b', help='balance', action='store_true')
    # parser.add_argument('-k', help='k', type=int, default=5)
    # parser.add_argument('-ds', help='data seed (-1 for time-dependent seed)', type=int, default=54321)
    # parser.add_argument('-ms', help='main seed (-1 for time-dependent seed)', type=int, default=12345)
    # parser.add_argument('-clip', help='clipping norm',type=float, default=4)
    # parser.add_argument('-ep', help='Epsilon for DP', type=float, default=1.0)
    # parser.add_argument('-dp', help='DP on', action='store_true', default=False)
    # parser.add_argument('--ts', help='Train size', type=float, default=0.3)
    # parser.add_argument('-c', help='CUDA num (-1 for CPU-only)', default='-1')
    # parser.add_argument('-clip', help='clipping norm',type=float, default=4)
    # parser.add_argument('-ep', help='Epsilon for DP', type=float, default=1.0)
    # parser.add_argument('-dp', help='DP on', action='store_true', default=False)
    
    # args = parser.parse_args()

    # if args.ds == -1: # seed for dataset generation
    #     seed_data = time.time()
    # else:
    #     seed_data = args.ds

    # if args.ms == -1: # seed for other random variables
    #     seed_main = time.time() + 1
    # else:
    #     seed_main = args.ms

    # start_time = time.time()
    # train_lfw(args.t, args.a, args.pi, args.pp, args.nw, args.nc, args.ni, args.van, args.b, args.k, args.ts, args.c,
    #           seed_data, seed_main,args)
    # duration = (time.time() - start_time)
    # logger.info("SGD ended in %0.2f hour (%.3f sec) " % (duration / float(3600), duration))
