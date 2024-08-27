import os
import torch
from torch.nn.functional import layer_norm as origin_layernorm
from torch.nn.functional import embedding as origin_embedding

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error
from .silent_fault_data import SilentFaultData, SilentFaultDataV2

__all__ = []


def _Singleton(cls):
    _instances = {}

    def _singleton(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return _singleton


@_Singleton
class _SilentFaultDetector:
    def __init__(self):
        self.silent_data_dict = dict()
        self.loss_scale = 1.0
        self.global_step = 0
        self.min_step = 7
        self.set_loss_scale_flag = False
        self.idx = None
        self.step = 0
        self.low_step = None
        self.high_step = None

    def set_asd_loss_scale(self, loss_scale=1.0):
        if loss_scale == 0:
            raise ValueError("loss scale cannot be 0." + pta_error(ErrCode.VALUE))
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

        if idx not in self.silent_data_dict:
            self.silent_data_dict[idx] = SilentFaultData()

        sfda = self.silent_data_dict[idx]
        
        if self.global_step <= self.min_step:
            self.step += 1
            self.global_step = self.step // (len(self.silent_data_dict) + 1)
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


_silent_fault_detector = _SilentFaultDetector()


def _patch_layernorm(input_layernorm, normalized_shape, *args, **kwargs):
    if input_layernorm is not None and input_layernorm.requires_grad and input_layernorm._backward_hooks is None:
        if "weight" in kwargs:
            input_layernorm.register_hook(_silent_fault_detector.silent_fault_check_hook(kwargs["weight"]))
        elif len(args) > 0:
            input_layernorm.register_hook(_silent_fault_detector.silent_fault_check_hook(args[0]))
    return origin_layernorm(input_layernorm, normalized_shape, *args, **kwargs)


def _patch_embedding(input_embedding, weight, *args, **kwargs):
    if weight is not None and weight.requires_grad and weight._backward_hooks is None:
        weight.register_hook(_silent_fault_detector.silent_fault_check_hook(weight))
    return origin_embedding(input_embedding, weight, *args, **kwargs)


def _asd_patch():
    env_value = os.getenv("NPU_ASD_ENABLE", "0")
    if env_value.isdigit() and int(env_value) and torch_npu._C._npu_support_silentClientV2():
        return

    if env_value not in ["0", "1"]:
        raise ValueError("NPU_ASD_ENABLE should be 0 or 1!" + pta_error(ErrCode.VALUE))

    if int(env_value):
        torch.nn.functional.layer_norm = _patch_layernorm
        torch.nn.functional.embedding = _patch_embedding


@_Singleton
class _SilentFaultDetectorV2:
    def __init__(self):
        self.silent_data_dict = dict()
        self.min_step = 100

    def silent_fault_check(self, idx, asd_enable, grad):
        if grad.dtype != torch.bfloat16 and grad.dtype != torch.float32:
            return

        val = torch.norm(grad)

        if idx not in self.silent_data_dict:
            self.silent_data_dict[idx] = SilentFaultDataV2()

        sfda = self.silent_data_dict[idx]

        torch_npu._npu_silent_check_v2(val, grad, sfda.check_tensor, sfda.step_tensor, self.min_step, sfda.upper_thresh[0],
                                       sfda.sigma_thresh[0], sfda.upper_thresh[1], sfda.sigma_thresh[1], asd_enable)


_silent_fault_detector_v2 = _SilentFaultDetectorV2()