# Copyright (c) 2026 Huawei Technologies Co., Ltd
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

"""
Add validation cases for torch.distributed.fsdp private APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some FSDP private APIs, so this file is added.
2. This file validates:
   torch.distributed.fsdp._fully_shard._fsdp_init._get_device_from_mesh
   torch.distributed.fsdp._runtime_utils._lazy_init
   (extendable)
"""

import os
import socket
import types
from functools import wraps

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import run_tests, TestCase


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class TestFSDPPrivateAPI(TestCase):
    MAIN_PROCESS_RANK = -1

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                for p in self.processes:
                    p.join()
                    if p.exitcode != 0:
                        self.fail(f"child process {p.name} exited with code {p.exitcode}")
            else:
                fn()

        return types.MethodType(wrapper, self)

    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def setUp(self):
        super(TestCase, self).setUp()
        if not torch.npu.is_available() or torch.npu.device_count() < self.world_size:
            self.skipTest(
                f"Multi-NPU {self.world_size} condition not satisfied"
            )
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_get_free_port())
        os.environ["BACKEND"] = dist.Backend.HCCL
        self.processes = []
        self.rank = self.MAIN_PROCESS_RANK
        proc = torch.multiprocessing.get_context("spawn").Process

        for rank in range(int(self.world_size)):
            process = proc(
                target=self.__class__._run,
                name="process " + str(rank),
                args=(rank, self._current_test_name()),
            )
            process.start()
            self.processes.append(process)

    def tearDown(self):
        super().tearDown()
        for p in self.processes:
            p.terminate()
        self.processes = []

    def _current_test_name(self) -> str:
        return self.id().split(".")[-1]

    @property
    def world_size(self) -> int:
        return 2

    @classmethod
    def _run(cls, rank: int, test_name: str) -> None:
        self = cls(test_name)
        self.rank = rank
        getattr(self, test_name)()

    def dist_init(self):
        torch.npu.set_device(self.rank)
        dist.init_process_group(backend="hccl", rank=self.rank, world_size=self.world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_get_device_from_mesh(self):
        """Get torch.device from a DeviceMesh (1D and 2D)."""
        from torch.distributed.fsdp._fully_shard._fsdp_init import _get_device_from_mesh

        self.dist_init()

        mesh = init_device_mesh("npu", (self.world_size,))
        dev = _get_device_from_mesh(mesh)
        self.assertIsInstance(dev, torch.device)
        self.assertEqual(dev.type, "npu")
        self.assertEqual(dev.index, self.rank)
        self.assertEqual(dev.index, torch.npu.current_device())

        mesh_2d = init_device_mesh("npu", (self.world_size, 1))
        dev_2d = _get_device_from_mesh(mesh_2d)
        self.assertIsInstance(dev_2d, torch.device)
        self.assertEqual(dev_2d.type, "npu")
        self.assertEqual(dev_2d.index, self.rank)

        mesh_cpu = init_device_mesh("cpu", (1,))
        dev_cpu = _get_device_from_mesh(mesh_cpu)
        self.assertIsInstance(dev_cpu, torch.device)
        self.assertEqual(dev_cpu.type, "cpu")
        self.assertIsNone(dev_cpu.index)
        dist.barrier()

    @skipIfUnsupportMultiNPU(2)
    def test_lazy_init(self):
        """Directly call _lazy_init(state, root_module) on root and non-root FSDP."""
        from torch.distributed.fsdp._runtime_utils import _lazy_init

        self.dist_init()

        lin1 = nn.Linear(8, 8).to(f"npu:{self.rank}")
        lin2 = nn.Linear(8, 8).to(f"npu:{self.rank}")
        sub_fsdp = FSDP(lin1, use_orig_params=True)
        root_fsdp = FSDP(nn.Sequential(sub_fsdp, lin2), use_orig_params=True)

        self.assertIsNone(root_fsdp._is_root)
        ret = _lazy_init(root_fsdp, root_fsdp)
        self.assertIs(ret, root_fsdp)
        self.assertTrue(root_fsdp._is_root)
        self.assertFalse(sub_fsdp._is_root)
        _lazy_init(root_fsdp, root_fsdp)
        self.assertTrue(root_fsdp._is_root)

        x = torch.randn(4, 8, device=f"npu:{self.rank}")
        out = root_fsdp(x)
        self.assertEqual(out.shape, torch.Size([4, 8]))
        self.assertEqual(out.device.type, "npu")

        self.assertIsNone(_lazy_init(sub_fsdp, sub_fsdp))
        self.assertFalse(sub_fsdp._is_root)
        dist.barrier()


if __name__ == "__main__":
    run_tests()
