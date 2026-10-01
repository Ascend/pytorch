# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
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
"""
Add validation cases for
torch.distributed.fsdp._common_utils._get_module_fsdp_state_if_fully_sharded_module
on NPU.

This file validates that a normal NPU module returns None and an FSDP-wrapped
NPU module returns the corresponding FSDP state.
"""

import socket

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_module_fsdp_state_if_fully_sharded_module,
)

from torch_npu.testing.testcase import TestCase, run_tests


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestGetModuleFSDPStateIfFullyShardedModule(TestCase):
    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()

    def test_normal_npu_module(self):
        torch.npu.set_device(0)
        device = torch.device("npu:0")

        module = nn.Linear(4, 4).to(device)

        result = _get_module_fsdp_state_if_fully_sharded_module(module)

        self.assertIsNone(result)

    def test_fsdp_npu_module(self):
        torch.npu.set_device(0)
        device = torch.device("npu:0")

        dist.init_process_group(
            backend="hccl",
            init_method=f"tcp://127.0.0.1:{find_free_port()}",
            rank=0,
            world_size=1,
        )

        module = nn.Linear(4, 4).to(device)
        fsdp_module = FSDP(module, device_id=device)

        result = _get_module_fsdp_state_if_fully_sharded_module(
            fsdp_module
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, _FSDPState)


if __name__ == "__main__":
    run_tests()
