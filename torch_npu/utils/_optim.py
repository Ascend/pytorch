from functools import partial, partialmethod, wraps

import torch
from torch.optim.sgd import SGD as SrcSGD
from torch.optim.sgd import sgd as src_sgd
from torch.optim.rprop import Rprop as SrcRprop
from torch.optim.rprop import rprop as src_rprop
from torch.optim.rmsprop import RMSprop as SrcRMSprop
from torch.optim.rmsprop import rmsprop as src_rmsprop
from torch.optim.radam import RAdam as SrcRAdam
from torch.optim.radam import radam as src_radam
from torch.optim.nadam import NAdam as SrcNAdam
from torch.optim.nadam import nadam as src_nadam
from torch.optim.asgd import ASGD as SrcASGD
from torch.optim.asgd import asgd as src_asgd
from torch.optim.adamw import AdamW as SrcAdamW
from torch.optim.adamw import adamw as src_adamw
from torch.optim.adamax import Adamax as SrcAdamax
from torch.optim.adamax import adamax as src_adamax
from torch.optim.adam import Adam as SrcAdam
from torch.optim.adam import adam as src_adam
from torch.optim.adagrad import Adagrad as SrcAdagrad
from torch.optim.adagrad import adagrad as src_adagrad
from torch.optim.adadelta import Adadelta as SrcAdadelta
from torch.optim.adadelta import adadelta as src_adadelta


def wrap_optim_warning_func(func, name):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not wrapper.warned:
            print(f"Warning: NPU does not support argument 'foreach' in torch.optim.{name}, "
                  f"we set foreach=False by default. Please do not set any value for this argument.")
            wrapper.warned = True
        return func(*args, **kwargs)
    wrapper.warned = False
    return wrapper


def partialclass(cls, *args, **kwargs):

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
        __name__ = cls.__name__

    NewCls.__name__ = cls.__name__

    return NewCls


SGD = partialclass(SrcSGD, foreach=False)
sgd = wrap_optim_warning_func(partial(src_sgd, foreach=False), 'sgd')

Rprop = partialclass(SrcRprop, foreach=False)
rprop = wrap_optim_warning_func(partial(src_rprop, foreach=False), 'rprop')

RMSprop = partialclass(SrcRMSprop, foreach=False)
rmsprop = wrap_optim_warning_func(partial(src_rmsprop, foreach=False), 'rmsprop')

RAdam = partialclass(SrcRAdam, foreach=False)
radam = wrap_optim_warning_func(partial(src_radam, foreach=False), 'radam')

NAdam = partialclass(SrcNAdam, foreach=False)
nadam = wrap_optim_warning_func(partial(src_nadam, foreach=False), 'nadam')

ASGD = partialclass(SrcASGD, foreach=False)
asgd = wrap_optim_warning_func(partial(src_asgd, foreach=False), 'asgd')

AdamW = partialclass(SrcAdamW, foreach=False)
adamw = wrap_optim_warning_func(partial(src_adamw, foreach=False), 'adamw')

Adamax = partialclass(SrcAdamax, foreach=False)
adamax = wrap_optim_warning_func(partial(src_adamax, foreach=False), 'adamax')

Adam = partialclass(SrcAdam, foreach=False)
adam = wrap_optim_warning_func(partial(src_adam, foreach=False), 'adam')

Adagrad = partialclass(SrcAdagrad, foreach=False)
adagrad = wrap_optim_warning_func(partial(src_adagrad, foreach=False), 'adagrad')

Adadelta = partialclass(SrcAdadelta, foreach=False)
adadelta = wrap_optim_warning_func(partial(src_adadelta, foreach=False), 'adadelta')


def add_optim_method():
    torch.optim.SGD = SGD
    torch.optim.sgd = sgd
    torch.optim.Rprop = Rprop
    torch.optim.rprop = rprop
    torch.optim.RMSprop = RMSprop
    torch.optim.rmsprop = rmsprop
    torch.optim.RAdam = RAdam
    torch.optim.radam = radam
    torch.optim.NAdam = NAdam
    torch.optim.nadam = nadam
    torch.optim.ASGD = ASGD
    torch.optim.asgd = asgd
    torch.optim.AdamW = AdamW
    torch.optim.adamw = adamw
    torch.optim.Adamax = Adamax
    torch.optim.adamax = adamax
    torch.optim.Adam = Adam
    torch.optim.adam = adam
    torch.optim.Adagrad = Adagrad
    torch.optim.adagrad = adagrad
    torch.optim.Adadelta = Adadelta
    torch.optim.adadelta = adadelta
