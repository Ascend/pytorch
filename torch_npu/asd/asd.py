import os
import torch
from torch.nn.functional import layer_norm as origin_layernorm
from torch.nn.functional import embedding as origin_embedding

import torch_npu
from .silent_fault_data import SilentFaultData


def Singleton(cls):
    _instances = {}

    def _singleton(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return _singleton


@Singleton
class SilentFaultDetector:
    def __init__(self):
        self.silent_data_dict = dict()
        self.loss_scale = 1.0
        self.global_step = 0
        self.min_step = 7
        self.set_loss_scale_flag = False
        self.idx = None
        self.dict_size = 0
        self.step = 0
        self.low_step = None
        self.high_step = None

    def set_asd_loss_scale(self, loss_scale=1.0):
        if loss_scale == 0:
            raise ValueError("loss scale cannot be 0.")
        self.set_loss_scale_flag = True
        self.loss_scale = loss_scale

    def silent_fault_check(self, grad):
        if self.low_step is None or self.high_step is None:
            self.low_step = torch.tensor(0, dtype=torch.int32).npu()
            self.high_step = torch.tensor(self.min_step, dtype=torch.int32).npu()
        if grad.dtype == torch.float16:
            if not self.set_loss_scale_flag:
                return 
            else:
                grad = grad.float() / self.loss_scale

        val = torch.norm(grad)
        idx = self.idx
        silent_data_dict = self.silent_data_dict

        if idx not in silent_data_dict:
            silent_data_dict[idx] = SilentFaultData()
            self.dict_size += 1

        sfda = silent_data_dict[idx]
        
        if self.global_step <= self.min_step:
            self.step += 1
            self.global_step = self.step // self.dict_size
            step_tensor = self.low_step
        else:
            step_tensor = self.high_step

        torch_npu._npu_silent_check(grad, val, sfda.pre_val, sfda.min_val, sfda.max_val, step_tensor, self.min_step,
                                    sfda.upper_thresh[0], sfda.sigma_thresh[0], sfda.upper_thresh[1], sfda.sigma_thresh[1])

    def silent_fault_check_hook(self, weight):
        def hook(grad):
            self.idx = id(weight)
            self.silent_fault_check(grad)
            return
        return hook


silent_fault_detector = SilentFaultDetector()


def custom_layernorm(input_layernorm, normalized_shape, weight, bias, eps):
    if input_layernorm is not None and input_layernorm.requires_grad and input_layernorm._backward_hooks is None:
        input_layernorm.register_hook(silent_fault_detector.silent_fault_check_hook(weight))
    return origin_layernorm(input_layernorm, normalized_shape, weight, bias, eps)


def custom_embedding(input_embedding, weight, padding_idx, max_norm,
            norm_type, scale_grad_by_freq, sparse):
    if weight is not None and weight.requires_grad and weight._backward_hooks is None:
        weight.register_hook(silent_fault_detector.silent_fault_check_hook(weight))
    return origin_embedding(input_embedding, weight, padding_idx, max_norm,
            norm_type, scale_grad_by_freq, sparse)


def asd_patch():
    torch.nn.functional.layer_norm = custom_layernorm
    torch.nn.functional.embedding = custom_embedding
