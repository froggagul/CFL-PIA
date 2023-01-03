from models import Model
from typing import List
from argparse import Namespace
import os
import torch

class ExperimentCheckpoint():
    def __init__(
        self,
        network_global: Model,
        cluster_networks: List[Model],
        worker_networks: List[Model],
        worker_networks_IFCA: List[Model],
        args: Namespace,
    ) -> None:
        self.network_global = network_global
        self.cluster_networks = cluster_networks
        self.worker_networks = worker_networks
        self.worker_networks_IFCA = worker_networks_IFCA
        self.args = args

    def _save_models(self, dir: os.PathLike, prefix: str, models: List[Model]):
        for index, model in enumerate(models):
            model.save(os.path.join(dir, f'{prefix}_{index}.pt'))

    def _load_models(self, dir: os.PathLike, prefix: str, models: List[Model]):
        for index, model in enumerate(models):
            model.load(os.path.join(dir, f'{prefix}_{index}.pt'))

    def save(self, dir: os.PathLike):
        if os.path.isdir(dir):
            raise Exception(f"directory {dir} exists")

        os.makedirs(dir, exist_ok=False)

        self.network_global.save(os.path.join(dir, 'network_global.pt'))
        self._save_models(dir, 'cluster_networks', self.cluster_networks)
        self._save_models(dir, 'worker_networks', self.worker_networks)
        self._save_models(dir, 'worker_networks_IFCA', self.worker_networks_IFCA)
        torch.save(self.args, os.path.join(dir, 'argument.pt'))

    def load(self, dir: os.PathLike):
        if not os.path.isdir(dir):
            raise Exception(f"directory {dir} not exists")
        self.network_global.load(os.path.join(dir, 'network_global.pt'))
        self._load_models(dir, 'cluster_networks', self.cluster_networks)
        self._load_models(dir, 'worker_networks', self.worker_networks)
        self._load_models(dir, 'worker_networks_IFCA', self.worker_networks_IFCA)
        self.args = torch.load(os.path.join(dir, 'argument.pt'))
