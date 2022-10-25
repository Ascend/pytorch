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

import threading
import torch
import torch._C
import torch._tensor_str
global print_holder
print_holder = '{}'

def generate_string_to_print(tuple_to_print):
    length = len(tuple_to_print)
    format_string = tuple_to_print[length - 1].decode()
    for i in range(length - 1):
        format_string = format_string.replace(print_holder, torch._tensor_str._tensor_str(tuple_to_print[i], 0), 1)
    return format_string

def print_deque_tensor():
    while True:
        tuple_to_print = torch._C._npu_deque_tensor()
        print(generate_string_to_print(tuple_to_print))

class NpuTensorManager(object):
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
            cls._instance.is_enter_npu_print = False
            cls._instance.npu_tensors_to_print = []
            cls._instance.start_deque_thread = False
        return cls._instance

    def __init__(self):
        pass
    
    def add_npu_tensor_to_print(self, npu_tensor):
        self.npu_tensors_to_print.append(npu_tensor)

    def get_npu_tensor_to_print(self):
        return self.npu_tensors_to_print

    def enter_npu_print(self):
        self.is_enter_npu_print = True
        self.npu_tensors_to_print = []
    
    def exit_npu_print(self):
        self.is_enter_npu_print = False
        self.npu_tensors_to_print = []
        if not self.start_deque_thread:
            self.start_deque_thread = True
            deque_thread = threading.Thread(target = print_deque_tensor)
            deque_thread.daemon = True
            deque_thread.start()

def npu_lazy_print(args):
    if not torch.npu.is_graph_mode():
        print(args)
    elif isinstance(args, torch.Tensor):
        torch.npu_enque_tensor([args], str(args))
    elif isinstance(args, list):
        for t in args:
            if not isinstance(t, torch.Tensor):
                raise RuntimeError("npu lazy_print only support tensor, "
                                   "tensor list or format string, while"
                                   "not support list of ", t.__class__.__name__)
        torch.npu_enque_tensor(args, str(args))
    elif isinstance(args, str):
        tm = NpuTensorManager()
        tensor_list = tm.get_npu_tensor_to_print()
        if len(tensor_list) == 0:
            print(args)
        if not len(tensor_list) == args.count(print_holder):
            raise RuntimeError("num of input npu tensor must be equal with"
                               "count of print holder")
        torch.npu_enque_tensor(tensor_list, args)
    else:
        raise RuntimeError("npu lazy_print only support tensor, "
                           "tensor list or format string, while"
                           "not support ", args.__class__.__name__)
    tm = NpuTensorManager()
    tm.exit_npu_print()

class GraphPrinter(object):
    def __init__(self):
        self.lazy_print = npu_lazy_print
    
    def __getattribute__(self, name):
        if name == "lazy_print":
            tm = NpuTensorManager()
            tm.enter_npu_print()
        return object.__getattribute__(self, name)

