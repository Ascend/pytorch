from functools import wraps

import torch
from torch.optim.sgd import sgd, SGD
from torch.optim.rprop import rprop, Rprop
from torch.optim.rmsprop import rmsprop, RMSprop
from torch.optim.radam import radam, RAdam
from torch.optim.nadam import nadam, NAdam
from torch.optim.asgd import asgd, ASGD
from torch.optim.adamw import adamw, AdamW
from torch.optim.adamax import adamax, Adamax
from torch.optim.adam import adam, Adam
from torch.optim.adagrad import adagrad, Adagrad
from torch.optim.adadelta import adadelta, Adadelta
import torch_npu

from .utils import should_print_warning


def is_support_foreach():
    device_name = torch_npu.npu.get_device_name(0)
    return device_name > "Ascend910B" and device_name < "Ascend910P"


def monkey_patch_optimizer(optimizer):
    @wraps(optimizer)
    def new_optimizer(*args, **kwargs):
        if not is_support_foreach():
            if should_print_warning():
                print(f"Warning: NPU does not support argument 'foreach' in this device type, "
                      f"we set foreach=False by default. Please do not set any value for this argument.")
            kwargs['foreach'] = False
        if 'foreach' in kwargs:
            kwargs['foreach'] = is_support_foreach() and kwargs['foreach']
        return optimizer(*args, **kwargs)
    
    return new_optimizer


def add_optim_method():
    torch.optim.SGD = monkey_patch_optimizer(SGD)
    torch.optim.sgd = monkey_patch_optimizer(sgd)
    torch.optim.Rprop = monkey_patch_optimizer(Rprop)
    torch.optim.rprop = monkey_patch_optimizer(rprop)
    torch.optim.RMSprop = monkey_patch_optimizer(RMSprop)
    torch.optim.rmsprop = monkey_patch_optimizer(rmsprop)
    torch.optim.RAdam = monkey_patch_optimizer(RAdam)
    torch.optim.radam = monkey_patch_optimizer(radam)
    torch.optim.NAdam = monkey_patch_optimizer(NAdam)
    torch.optim.nadam = monkey_patch_optimizer(nadam)
    torch.optim.ASGD = monkey_patch_optimizer(ASGD)
    torch.optim.asgd = monkey_patch_optimizer(asgd)
    torch.optim.AdamW = monkey_patch_optimizer(AdamW)
    torch.optim.adamw = monkey_patch_optimizer(adamw)
    torch.optim.Adamax = monkey_patch_optimizer(Adamax)
    torch.optim.adamax = monkey_patch_optimizer(adamax)
    torch.optim.Adam = monkey_patch_optimizer(Adam)
    torch.optim.adam = monkey_patch_optimizer(adam)
    torch.optim.Adagrad = monkey_patch_optimizer(Adagrad)
    torch.optim.adagrad = monkey_patch_optimizer(adagrad)
    torch.optim.Adadelta = monkey_patch_optimizer(Adadelta)
    torch.optim.adadelta = monkey_patch_optimizer(adadelta)
