import os
import warnings
import torch
from torch.nn import Module

import torch_npu
from torch_npu.utils.error_code import ErrCode, pta_error
from torch_npu.asd.asd import _silent_fault_detector_v2


original_call = Module.__call__
IS_IN_BACKWARD = 0


def input_hook(idx, asd_flag):
    def hook(grad):
        global IS_IN_BACKWARD

        if idx != "":
            IS_IN_BACKWARD = IS_IN_BACKWARD & 1  # 011 & 001 = 001
            _silent_fault_detector_v2.silent_fault_check(idx, asd_flag, grad)
        else:
            IS_IN_BACKWARD = IS_IN_BACKWARD & 2  # 011 & 010 = 010

        if not IS_IN_BACKWARD:
            torch_npu._C._npu_set_call_state("forward")
        return
    return hook


def output_hook(grad):
    global IS_IN_BACKWARD
    IS_IN_BACKWARD = 3  # 011
    torch_npu._C._npu_set_call_state("backward")
    return grad


def _is_inner_module(module):
    return len(module._modules) == 0


class SilentCheckState:
    def __init__(self):
        self.init_param()
        self.init_marks = {}
        self.weight_hook_flags = {}
        self.last_weight_hook_flags = {}

    def init_param(self):
        self.first_forward = True
        self.input_hook_flag = False
        self.is_training = False
        self.first_module_id = ""
        self.first_weight = None
        self.last_weight = None
        self.last_tensor = None
        self.last_tensor_id = None
        self.first_tensor_id = None

    def init_module_info(self, module_id, training):
        self.first_module_id = module_id
        self.first_forward = False
        self.is_training = training
        if self.is_training:
            torch_npu._C._npu_set_module_train_state("train")
        else:
            torch_npu._C._npu_set_module_train_state("infer")

    def search_first_weight(self, module):
        # Search the first weight
        if not self.init_marks.get(self.first_module_id, False) and self.first_weight is None:
            for param_name, param in module._parameters.items():
                if isinstance(param, torch.Tensor) and param.requires_grad:
                    self.first_weight = param
                    break

    def register_input_hook_before_call(self, asd_flag, *args):
        # Search the first tensor (if the first tensor is input)
        if self.is_training and not self.input_hook_flag:
            for x in args:
                if isinstance(x, torch.Tensor) and x.requires_grad:
                    x.register_hook(input_hook(self.first_module_id, asd_flag))
                    self.input_hook_flag = True
                    break

    def register_input_hook_after_call(self, output):
        # Search the first tensor (if the first tensor is output of an inner module)
        if not self.input_hook_flag:
            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.register_hook(input_hook(self.first_module_id, asd_enable))
                self.input_hook_flag = True
                self.first_tensor_id = id(output)

    def search_last_weight(self, module):
        # Search the last weight (only in inner module)
        if not self.init_marks.get(self.first_module_id, False) and _is_inner_module(module):
            for param_name, param in module._parameters.items():
                if isinstance(param, torch.Tensor) and param.requires_grad:
                    self.last_weight = param

    def search_last_tensor(self, output):
        # Search the last tensor
        if isinstance(output, torch.Tensor) and output.requires_grad:
            self.last_tensor_id = id(output)
            self.last_tensor = output

    def init_all_hook(self, asd_flag):
        if self.is_training:
            # Otherwise, there is only one weight in the outer module
            if self.first_tensor_id != self.last_tensor_id:
                if self.last_tensor is not None:
                    self.last_tensor.register_hook(output_hook)
                if not self.last_weight_hook_flags.get(self.first_module_id, False):
                    if self.last_weight is not None:
                        self.last_weight.register_hook(output_hook)
                        self.last_weight_hook_flags[self.first_module_id] = True
                if not self.weight_hook_flags.get(self.first_module_id, False):
                    if self.first_weight is not None:
                        self.first_weight.register_hook(input_hook("", asd_flag))
                        self.weight_hook_flags[self.first_module_id] = True
                self.init_marks[self.first_module_id] = True

silent_check = SilentCheckState()
asd_enable = 0


def _custom_call(self, *args, **kwargs): 
    global asd_enable
    global silent_check
    global IS_IN_BACKWARD

    if not torch.npu.is_initialized():
        return original_call(self, *args, **kwargs)

    if not IS_IN_BACKWARD:
        if silent_check.first_forward:
            silent_check.init_module_info(id(self), self.training)
            self.outer = True

        # Search the first tensor (if the first tensor is input)
        silent_check.register_input_hook_before_call(asd_enable, *args)

    tmp = original_call(self, *args, **kwargs)

    if silent_check.is_training and not IS_IN_BACKWARD:
        # Search the first weight
        silent_check.search_first_weight(self)

        # Search the first tensor (if the first tensor is output of an inner module)
        silent_check.register_input_hook_after_call(tmp)

        # Search the last weight (only in inner module)
        silent_check.search_last_weight(self)
        
        # Search the last tensor
        silent_check.search_last_tensor(tmp)

    if not IS_IN_BACKWARD:
        if hasattr(self, "outer") and self.outer:
            silent_check.init_all_hook(asd_enable)
            silent_check.init_param()
            self.outer = False

    return tmp


def add_asd_patch():
    global asd_enable

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
