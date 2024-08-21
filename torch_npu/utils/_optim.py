from functools import partial, partialmethod, wraps

import torch
from torch.optim.sgd import sgd as src_sgd
from torch.optim.rprop import rprop as src_rprop
from torch.optim.rmsprop import rmsprop as src_rmsprop
from torch.optim.radam import radam as src_radam
from torch.optim.nadam import nadam as src_nadam
from torch.optim.asgd import asgd as src_asgd
from torch.optim.adamw import adamw as src_adamw
from torch.optim.adamax import adamax as src_adamax
from torch.optim.adam import adam as src_adam
from torch.optim.adagrad import adagrad as src_adagrad
from torch.optim.adadelta import adadelta as src_adadelta

from .utils import should_print_warning


def wrap_optim_warning_func(func, name):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not wrapper.warned and should_print_warning():
            print(f"Warning: NPU does not support argument 'foreach' in torch.optim.{name}, "
                  f"we set foreach=False by default. Please do not set any value for this argument.")
            wrapper.warned = True
        return func(*args, **kwargs)
    wrapper.warned = False
    return wrapper


def partial_class(cls, *args, **kwargs):
    cls.__init__ = partialmethod(cls.__init__, *args, **kwargs)


sgd = wrap_optim_warning_func(partial(src_sgd, foreach=False), 'sgd')
rprop = wrap_optim_warning_func(partial(src_rprop, foreach=False), 'rprop')
rmsprop = wrap_optim_warning_func(partial(src_rmsprop, foreach=False), 'rmsprop')
radam = wrap_optim_warning_func(partial(src_radam, foreach=False), 'radam')
nadam = wrap_optim_warning_func(partial(src_nadam, foreach=False), 'nadam')
asgd = wrap_optim_warning_func(partial(src_asgd, foreach=False), 'asgd')
adamw = wrap_optim_warning_func(partial(src_adamw, foreach=False), 'adamw')
adamax = wrap_optim_warning_func(partial(src_adamax, foreach=False), 'adamax')
adam = wrap_optim_warning_func(partial(src_adam, foreach=False), 'adam')
adagrad = wrap_optim_warning_func(partial(src_adagrad, foreach=False), 'adagrad')
adadelta = wrap_optim_warning_func(partial(src_adadelta, foreach=False), 'adadelta')


def add_optim_method():
    partial_class(torch.optim.SGD, foreach=False)
    torch.optim.sgd = sgd
    partial_class(torch.optim.Rprop, foreach=False)
    torch.optim.rprop = rprop
    partial_class(torch.optim.RMSprop, foreach=False)
    torch.optim.rmsprop = rmsprop
    partial_class(torch.optim.RAdam, foreach=False)
    torch.optim.radam = radam
    partial_class(torch.optim.NAdam, foreach=False)
    torch.optim.nadam = nadam
    partial_class(torch.optim.ASGD, foreach=False)
    torch.optim.asgd = asgd
    partial_class(torch.optim.AdamW, foreach=False)
    torch.optim.adamw = adamw
    partial_class(torch.optim.Adamax, foreach=False)
    torch.optim.adamax = adamax
    partial_class(torch.optim.Adam, foreach=False)
    torch.optim.adam = adam
    partial_class(torch.optim.Adagrad, foreach=False)
    torch.optim.adagrad = adagrad
    partial_class(torch.optim.Adadelta, foreach=False)
    torch.optim.adadelta = adadelta
