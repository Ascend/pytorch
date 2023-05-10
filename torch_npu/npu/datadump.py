# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

import os
import threading
from typing import List

import numpy
from torch import Tensor
from torch.autograd.function import Function

import torch_npu


class DatadumpBeginOp(Function):
    @staticmethod
    def forward(ctx, tensor, ops, dumpBackward, capacity):
        ctx.npu_dump_backward = dumpBackward
        torch_npu._C._npu_datadump_enable(ops, capacity)
        return tensor

    @staticmethod
    def backward(ctx, grad_outputs):
        if ctx.npu_dump_backward:
            torch_npu._C._npu_datadump_disable()
        return grad_outputs, None, None, None


class DatadumpEndOp(Function):
    @staticmethod
    def forward(ctx, tensor, ops, dumpBackward, capacity):
        ctx.npu_dump_backward = dumpBackward
        ctx.npu_dump_ops = ops
        ctx.capacity = capacity
        torch_npu._C._npu_datadump_disable()
        return tensor

    @staticmethod
    def backward(ctx, grad_outputs):
        if ctx.npu_dump_backward:
            torch_npu._C._npu_datadump_enable(ctx.npu_dump_ops, ctx.capacity)
        return grad_outputs, None, None, None


class NpuDatadumpMgr(object):
    def __init__(self):
        self._dequeThread = None
        self._dumpBackward = False
        self._ops = []
        self._path = None
        self._status = False
        self._deviceId = -1
        self.capacity = None

    def isDumpBackward(self):
        return self._dumpBackward

    def getOps(self):
        return self._ops

    def enable(self, path, ops, dumpBackward):
        if self._status:
            raise RuntimeError("before enable datadump, please disable first",
                               self.__class__.__name__)
        self._status = True
        self._path = path
        self._ops = ops
        self._dumpBackward = dumpBackward
        if not self._dequeThread:
            self.__datadump()

    def disable(self):
        if not self._status:
            raise RuntimeError("before disable datadump, please enable first",
                               self.__class__.__name__)
        self._status = False

    def __datadump(self):
        self._deviceId = torch_npu._C._npu_getDevice()
        print("Start datadump deque thread. device id: " + str(self._deviceId))
        self._dequeThread = threading.Thread(target=self.__dequeAndDump)
        self._dequeThread.daemon = True
        self._dequeThread.start()

    def __dequeAndDump(self):
        torch_npu._C._npu_setDevice(self._deviceId)
        while True:
            try:
                tensorTuple = torch_npu._C._npu_deque_tensor()
                self.__dump(tensorTuple)
            except Exception as e:
                print("datadump deque thread exception", e)

    def __dump(self, tensorTuple):
        length = len(tensorTuple)
        info = str(tensorTuple[length - 1].decode())
        infos = info.split('|')
        opName = infos[0]
        for i in range(length - 1):
            t = tensorTuple[i].numpy()
            metas = infos[i + 1].split('#')
            savePath = self._path + opName + str(i) + '_shape' + metas[0].replace(' ', '') \
                       + '_stride' + metas[1].replace(' ', '') + '_offset[' + metas[2] + \
                       ']_format[' + metas[3] + '].npy'
            numpy.save(savePath, t)
        print("Datadump: " + opName)


mgr = NpuDatadumpMgr()


def dump_enable(tensor: Tensor, path: str = "./", ops: List[str] = None,
                dump_backward: bool = False, capacity: int = 3):
    if torch_npu.npu.is_graph_mode():
        raise RuntimeError("datadump not support graph mode")
    if not path.endswith("/"):
        path = path + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    if ops is None:
        ops = []
    if dump_backward and not tensor.requires_grad:
        raise RuntimeError("Before enable dump_backward, please ensure dump_enable input tensor requires_grad is true.")
    mgr.enable(path, ops, dump_backward)
    if capacity < 3 or capacity > 2048:
        raise RuntimeError("capacity range [3, 2048].")
    mgr.capacity = capacity
    return DatadumpBeginOp.apply(tensor, mgr.getOps(), mgr.isDumpBackward(), mgr.capacity)


def dump_disable(tensor: Tensor):
    mgr.disable()
    return DatadumpEndOp.apply(tensor, mgr.getOps(), mgr.isDumpBackward(), mgr.capacity)
