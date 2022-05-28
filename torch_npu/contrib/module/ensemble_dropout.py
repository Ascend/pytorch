# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import List, Optional

import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

                        
class DropOutTask:
    def __init__(self, shape, dtype, device, p):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.p = p
        self.request_count = 0
        self.mask_queue = []
                
                        
class NpuFairseqDropout(torch.nn.Dropout):
    r"""FairseqDropout using on npu device

    Reference implementation link:
    https://github.com/facebookresearch/fairseq/blob/e0884db9a7ce83670e21af39bf785b616ce5e3e3/fairseq/modules/fairseq_dropout.py#L16


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
            raise RuntimeError("input type error!")

        if self.p == 0:
            return return_obj
        key = (shape, dtype, device, self.p)
        if key not in NpuFairseqDropout.task_dict:
            dropout_task = DropOutTask(shape, dtype, device, self.p)
            dropout_task.request_count += 1
            NpuFairseqDropout.task_dict[key] = dropout_task
            return return_obj
        elif not NpuFairseqDropout.task_dict[key].mask_queue:
            NpuFairseqDropout.task_dict[key].request_count += 1
            return return_obj
        else:
            mask, event = NpuFairseqDropout.task_dict[key].mask_queue.pop(0)
            if do_mask_flag:
                return torch.npu_dropout_do_mask(x, mask, self.p)[0]
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
                            mask = torch.npu_dropout_gen_mask(task.shape, p=task.p, dtype=task.dtype,
                                                                device=task.device)
                            event = None
                            task.mask_queue.append((mask, event))
            return hook_function

        model.register_forward_hook(mask_gen_hook_func())