# Copyright (c) 2022, Huawei Technologies.All rights reserved.
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

import os
import warnings
import logging as logger
from functools import wraps
import torch
import torch_npu

warnings.filterwarnings(action='once')


torch_fn_white_list = ['logspace', 'randint', 'hann_window', 'rand', 'full_like', 'ones_like', 'rand_like', 'randperm',
                       'arange', 'frombuffer', 'normal', '_empty_per_channel_affine_quantized', 'empty_strided',
                       'empty_like', 'scalar_tensor', 'tril_indices', 'bartlett_window', 'ones', 'sparse_coo_tensor',
                       'randn', 'kaiser_window', 'tensor', 'triu_indices', 'as_tensor', 'zeros', 'randint_like', 'full',
                       'eye', '_sparse_csr_tensor_unsafe', 'empty', '_sparse_coo_tensor_unsafe', 'blackman_window',
                       'zeros_like', 'range', 'sparse_csr_tensor', 'randn_like', 'from_file',
                       '_cudnn_init_dropout_state', '_empty_affine_quantized', 'linspace', 'hamming_window',
                       'empty_quantized', '_pin_memory', 'autocast', 'load', "Generator"]
torch_tensor_fn_white_list = ['new_empty', 'new_empty_strided', 'new_full', 'new_ones', 'new_tensor', 'new_zeros', 'to']
torch_module_fn_white_list = ['to', 'to_empty']
torch_cuda_fn_white_list = [
    'get_device_properties', 'get_device_name', 'get_device_capability', 'list_gpu_processes', 'set_device',
    'synchronize', 'mem_get_info', 'memory_stats', 'memory_summary', 'memory_allocated', 'max_memory_allocated',
    'reset_max_memory_allocated', 'memory_reserved', 'max_memory_reserved', 'reset_max_memory_cached',
    'reset_peak_memory_stats', 'device'
]
torch_profiler_fn_white_list = ['profile']
torch_distributed_fn_white_list = ['__init__']
device_kwargs_list = ['device', 'device_type', 'map_location']


def wrapper_cuda(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        if args:
            args_new = list(args)
            args = replace_cuda_to_npu_in_list(args_new)
        if kwargs:
            for device_arg in device_kwargs_list: 
                device = kwargs.get(device_arg, None)
                if device is not None:
                    replace_cuda_to_npu_in_kwargs(kwargs, device_arg, device)     
            if 'experimental_config' in kwargs.keys() and not isinstance(kwargs.get('experimental_config'),
                                                                         torch_npu.profiler._ExperimentalConfig):
                logger.warning(
                    'The parameter experimental_config of torch.profiler.profile has been deleted by the tool '
                    'because it can only be used in cuda, please manually modify the code '
                    'and use the experimental_config parameter adapted to npu.')
                del kwargs['experimental_config']
            device_ids = kwargs.get('device_ids', None)
            if isinstance(device_ids, list):
                device_ids = replace_cuda_to_npu_in_list(device_ids)
        return fn(*args, **kwargs)

    return decorated


def replace_cuda_to_npu_in_kwargs(kwargs, device_arg, device):
    if isinstance(device, str) and 'cuda' in device:
        kwargs[device_arg] = device.replace('cuda', 'npu')
    elif (isinstance(device, torch.device) or str(type(device)) == "<class 'torch.device'>") and 'cuda' in device.type:
        device_info = 'npu:{}'.format(device.index) if device.index is not None else 'npu'
        kwargs[device_arg] = torch.device(device_info)
    elif isinstance(device, dict):
        kwargs[device_arg] = replace_cuda_to_npu_in_dict(device)


def replace_cuda_to_npu_in_list(args_list):
    for idx, arg in enumerate(args_list):
        if isinstance(arg, str) and 'cuda' in arg:
            args_list[idx] = arg.replace('cuda', 'npu')
        elif (isinstance(arg, torch.device) or str(type(arg)) == "<class 'torch.device'>") and 'cuda' in arg.type:
            device_info = 'npu:{}'.format(arg.index) if arg.index is not None else 'npu'
            args_list[idx] = torch.device(device_info)
        elif isinstance(arg, dict):
            args_list[idx] = replace_cuda_to_npu_in_dict(arg)
    return args_list


def replace_cuda_to_npu_in_dict(device_dict):
    new_dict = {}
    for key, value in device_dict.items():
        if isinstance(key, str):
            key = key.replace('cuda', 'npu')
        if isinstance(value, str):
            value = value.replace('cuda', 'npu')
        new_dict[key] = value
    return new_dict


def device_wrapper(enter_fn, white_list):
    for fn_name in white_list:
        fn = getattr(enter_fn, fn_name, None)
        if fn:
            setattr(enter_fn, fn_name, wrapper_cuda(fn))


def wrapper_hccl(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        if args:
            args_new = list(args)
            for idx, arg in enumerate(args_new):
                if isinstance(arg, str) and 'nccl' in arg:
                    args_new[idx] = arg.replace('nccl', 'hccl')
            args = args_new
        if kwargs:
            if isinstance(kwargs.get('backend', None), str):
                kwargs['backend'] = 'hccl'
        return fn(*args, **kwargs)

    return decorated


def patch_cuda():
    patchs = [
        ['cuda', torch_npu.npu], ['cuda.amp', torch_npu.npu.amp],
        ['cuda.random', torch_npu.npu.random],
        ['cuda.amp.autocast_mode', torch_npu.npu.amp.autocast_mode],
        ['cuda.amp.common', torch_npu.npu.amp.common],
        ['cuda.amp.grad_scaler', torch_npu.npu.amp.grad_scaler]
    ]
    torch_npu._apply_patches(patchs)


def patch_profiler():
    patchs = [
        ['profiler.profile', torch_npu.profiler.profile], 
        ['profiler.schedule', torch_npu.profiler.schedule],
        ['profiler.tensorboard_trace_handler', torch_npu.profiler.tensorboard_trace_handler],
        ['profiler.ProfilerAction', torch_npu.profiler.ProfilerAction],
        ['profiler.ProfilerActivity.CUDA', torch_npu.profiler.ProfilerActivity.NPU],
        ['profiler.ProfilerActivity.CPU', torch_npu.profiler.ProfilerActivity.CPU]
    ]
    torch_npu._apply_patches(patchs)


def warning_fn(msg, rank0=True):
    is_distributed = torch.distributed.is_available() and \
                     torch.distributed.is_initialized() and \
                     torch.distributed.get_world_size() > 1
    env_rank = os.getenv('RANK', None)

    if rank0 and is_distributed:
        if torch.distributed.get_rank() == 0:
            warnings.warn(msg, ImportWarning)
    elif rank0 and env_rank:
        if env_rank == '0':
            warnings.warn(msg, ImportWarning)
    else:
        warnings.warn(msg, ImportWarning)


def init():
    warning_fn('''
    *************************************************************************************************************
    The torch.Tensor.cuda and torch.nn.Module.cuda are replaced with torch.Tensor.npu and torch.nn.Module.npu now..
    The torch.cuda.DoubleTensor is replaced with torch.npu.FloatTensor cause the double type is not supported now..
    The backend in torch.distributed.init_process_group set to hccl now..
    The torch.cuda.* and torch.cuda.amp.* are replaced with torch.npu.* and torch.npu.amp.* now..
    The device parameters have been replaced with npu in the function below:
    {}
    *************************************************************************************************************
    '''.format(', '.join(
        ['torch.' + i for i in torch_fn_white_list] + ['torch.Tensor.' + i for i in torch_tensor_fn_white_list] +
        ['torch.nn.Module.' + i for i in torch_module_fn_white_list]))
    )

    # torch.cuda.*
    patch_cuda()
    device_wrapper(torch.cuda, torch_cuda_fn_white_list)

    # torch.profiler.*
    patch_profiler()
    device_wrapper(torch.profiler, torch_profiler_fn_white_list)

    # torch.*
    device_wrapper(torch, torch_fn_white_list)

    # torch.Tensor.*
    device_wrapper(torch.Tensor, torch_tensor_fn_white_list)
    torch.Tensor.cuda = torch.Tensor.npu
    torch.Tensor.is_cuda = torch.Tensor.is_npu
    torch.cuda.DoubleTensor = torch.npu.FloatTensor

    # torch.nn.Module.*
    device_wrapper(torch.nn.Module, torch_module_fn_white_list)
    torch.nn.Module.cuda = torch.nn.Module.npu

    # torch.distributed.init_process_group
    torch.distributed.init_process_group = wrapper_hccl(torch.distributed.init_process_group)
    torch.distributed.is_nccl_available = torch_npu.distributed.is_hccl_available
    torch.distributed.distributed_c10d.broadcast_object_list = torch_npu.distributed.distributed_c10d.broadcast_object_list
    torch.distributed.distributed_c10d.all_gather_object = torch_npu.distributed.distributed_c10d.all_gather_object

    # torch.nn.parallel.DistributedDataParallel
    device_wrapper(torch.nn.parallel.DistributedDataParallel, torch_distributed_fn_white_list)

    torch.npu.amp.autocast_mode.npu_autocast.__init__ = wrapper_cuda(torch.npu.amp.autocast_mode.npu_autocast.__init__)


init()
