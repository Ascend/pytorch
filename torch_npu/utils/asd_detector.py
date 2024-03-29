import torch
import torch_npu
from torch_npu.asd.asd import silent_fault_detector


def set_asd_loss_scale(loss_scale=1.0):
    silent_fault_detector.set_asd_loss_scale(loss_scale)
    return


def register_asd_hook(x, weight):
    if x is not None and x.requires_grad and x._backward_hooks is None:
        x.register_hook(silent_fault_detector.silent_fault_check_hook(weight))
    return
