import os
import warnings
import torch
from torch.nn import Module

import torch_npu
from torch_npu.utils.error_code import ErrCode, pta_error


original_call = Module.__call__

first_forward = True
input_hook_flag = True
weight_hook_flag = True


def input_hook(grad):
    torch_npu._C._npu_set_call_state("forward")
    return grad


def output_hook(grad):
    torch_npu._C._npu_set_call_state("backward")
    return grad


def _custom_call(self, *args, **kwargs):    
    global perf_dump_enable
    global perf_dump_state

    global asd_enable
    global first_forward
    global input_hook_flag
    global weight_hook_flag

    if not torch.npu.is_initialized():
        return original_call(self, *args, **kwargs)

    if input_hook_flag:
        for x in args:
            if isinstance(x, torch.Tensor):
                if x.requires_grad:
                    x.register_hook(input_hook)
                    input_hook_flag = False
                    break

    if weight_hook_flag:
        for param_name, param in self._parameters.items():
            if isinstance(param, torch.Tensor):
                if param.requires_grad:
                    param.register_hook(input_hook)
                    weight_hook_flag = False
                    break

    if first_forward:
        first_forward = False
        self.outer = True
        if self.training:
            torch_npu._C._npu_set_module_train_state("train")
        else:
            torch_npu._C._npu_set_module_train_state("infer")

    tmp = original_call(self, *args, **kwargs)

    if hasattr(self, "outer") and self.outer:
        if isinstance(tmp, torch.Tensor):
            if tmp.requires_grad:
                tmp.register_hook(output_hook)
        input_hook_flag = True
        weight_hook_flag = True
        first_forward = True
        self.outer = False

    return tmp


def add_asd_patch():
    asd_value = os.getenv("NPU_ASD_ENABLE", "0")
    if asd_value not in ["0", "1", "2", "3"]:
        raise ValueError("NPU_ASD_ENABLE should be 0, 1, 2 or 3. For details, 0 as `ASD closed`, "
                         "1 as `ASD opened, print error logs` "
                         "2 as `ASD opened, print error logs and raise exception`, "
                         "3 as `ASD opened, print debug logs and raise exception`" + pta_error(ErrCode.VALUE))
    asd_enable = int(asd_value)
    if asd_enable and not torch_npu._C._npu_support_silentClientV2():        
        warnings.warn(f"Warning: CANN version lower than 8.0.RC3 and currently does not support silent check 2.0 version. It will switch to 1.0 version.")
        asd_enable = 0

    if asd_enable:
        Module.__call__ = _custom_call
