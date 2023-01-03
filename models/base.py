import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Iterator, Tuple
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self):
        raise NotImplementedError("Model is interface, use implementation of Model")

    def get_params(self): # , clone: bool
        # TODO - check clone param exists
        params = OrderedDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                params[name] = param
        return params

    def set_params(self, params: Iterator[Tuple[str, Parameter]]):
        # TODO - set params
        pass

    def get_grads(self):
        # TODO - get grads
        pass

    def set_grads(self):
        # TODO - set grads
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class ModelFactory:
    registry = {}

    @classmethod
    def register(cls, name):
        import warnings
        def inner_wrapper(wrapped_class: Model):
            if name in cls.registry:
                warnings.warn(f'Register {name} already exists. Will replace it')
            cls.registry[name] = wrapped_class
        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> Model:
        exec_class = cls.registry[name]
        executer = exec_class(**kwargs)
        return executer
