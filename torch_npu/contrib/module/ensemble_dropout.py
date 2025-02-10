import logging
from typing import List, Optional
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_npu
from torch_npu.utils._error_code import ErrCode, ops_error
from torch_npu.contrib.module._ensemble_dropout import NpuPreGenDropout

logger = logging.getLogger(__name__)

__all__ = [
    "NpuCachedDropout",
    "NpuFairseqDropout"
]


class _DropOutTask:
    def __init__(self, shape, dtype, device, p):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.p = p
        self.request_count = 0
        self.mask_queue = []
                
                        
class NpuCachedDropout(torch.nn.Dropout):
    r"""FairseqDropout using on npu device

    .. note::
        Dynamic shapes are not supported.

    Args:
        p (float): probability of an element to be zeroed.
        module_name (string): the name of the model
    """
    
    task_dict = {}
    dropout_stream = None

    def __init__(self, p, module_name=None):
        super().__init__(p)
        self.module_name = module_name

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            shape = x.shape
            dtype = x.dtype
            device = x.device
            do_mask_flag = True
            return_obj = x
        elif isinstance(x, list):
            shape, dtype, device = x
            do_mask_flag = False
            return_obj = None
        else:
            raise RuntimeError("input type error!" + ops_error(ErrCode.TYPE))

        if self.p == 0:
            return return_obj
        key = (shape, dtype, device, self.p)
        if key not in NpuCachedDropout.task_dict:
            dropout_task = _DropOutTask(shape, dtype, device, self.p)
            dropout_task.request_count += 1
            NpuCachedDropout.task_dict[key] = dropout_task
            return return_obj
        elif not NpuCachedDropout.task_dict[key].mask_queue:
            NpuCachedDropout.task_dict[key].request_count += 1
            return return_obj
        else:
            mask, event = NpuCachedDropout.task_dict[key].mask_queue.pop(0)
            if do_mask_flag:
                return torch_npu.npu_dropout_do_mask(x, mask, self.p)[0]
            else:
                return mask

    @classmethod
    def enable_dropout_ensemble(cls, model):
        if cls.dropout_stream is None:
            cls.dropout_stream = torch.npu.Stream()

        def wait_stream_hook_func():
            def hook_function(module, inputs):
                torch.npu.current_stream().wait_stream(cls.dropout_stream)
            return hook_function
        model.register_forward_pre_hook(wait_stream_hook_func())

        def mask_gen_hook_func():
            def hook_function(module, inputs, outputs):
                for _, task in cls.task_dict.items():
                    if len(task.mask_queue) < task.request_count:
                        for j in range(task.request_count - len(task.mask_queue)):
                            mask = torch_npu.npu_dropout_gen_mask(task.shape, p=task.p, dtype=task.dtype,
                                                                  device=task.device)
                            event = None
                            task.mask_queue.append((mask, event))
            return hook_function

        model.register_forward_hook(mask_gen_hook_func())


NpuFairseqDropout = NpuCachedDropout

