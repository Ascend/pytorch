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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for torch.distributed.distributed_c10d.BroadcastOptions on NPU.

Covers binding identity, default field values, field read/write, and invalid
assignment for every exposed field. The two-rank HCCL case consumes the
options object and cleans up each worker's process group before returning.
Extend coverage by adding test methods for further option combinations or
tensor configurations.
"""

import os
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import BroadcastOptions

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestBroadcastOptions(TestCase):

    def test_binding_identity(self):
        from torch._C._distributed_c10d import BroadcastOptions as CBindingOptions
        from torch.distributed import BroadcastOptions as PackageOptions

        self.assertIs(CBindingOptions, BroadcastOptions)
        self.assertIs(PackageOptions, BroadcastOptions)
        self.assertEqual(BroadcastOptions.__module__, "torch.distributed.distributed_c10d")

    def test_default_field_values(self):
        options = BroadcastOptions()

        self.assertEqual(options.rootRank, 0)
        self.assertEqual(options.rootTensor, 0)
        self.assertEqual(options.timeout, timedelta(milliseconds=-1))
        self.assertTrue(options.asyncOp)

    def test_fields_read_write(self):
        options = BroadcastOptions()
        options.rootRank = 3
        options.rootTensor = 2
        options.timeout = timedelta(seconds=5)
        options.asyncOp = False

        self.assertEqual(options.rootRank, 3)
        self.assertEqual(options.rootTensor, 2)
        self.assertEqual(options.timeout, timedelta(seconds=5))
        self.assertFalse(options.asyncOp)

    def test_invalid_field_assignment_raises(self):
        options = BroadcastOptions()

        with self.assertRaises(TypeError):
            options.rootRank = "1"
        with self.assertRaises(TypeError):
            options.rootTensor = None
        with self.assertRaises(TypeError):
            options.timeout = "invalid"
        with self.assertRaises(TypeError):
            options.asyncOp = "not_bool"

    @staticmethod
    def _hccl_broadcast_worker(rank, world_size, result_dir):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29511'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)

        group = dist.distributed_c10d._get_default_group()
        tensor = torch.tensor([float(rank)], dtype=torch.float32).to(device_type)

        options = BroadcastOptions()
        options.rootRank = 1
        options.rootTensor = 0
        options.timeout = timedelta(seconds=30)
        options.asyncOp = True
        work = group.broadcast([tensor], options)
        work.wait()
        async_result = tensor.cpu().clone()

        tensor.copy_(torch.tensor([float(rank)], dtype=torch.float32).to(device_type))
        options_sync = BroadcastOptions()
        options_sync.rootRank = 0
        options_sync.rootTensor = 0
        options_sync.timeout = timedelta(seconds=30)
        options_sync.asyncOp = False
        work = group.broadcast([tensor], options_sync)
        work.wait()
        torch.save({"async_result": async_result, "sync_result": tensor.cpu()},
                   os.path.join(result_dir, f"rank{rank}_broadcast.pt"))
        dist.destroy_process_group()

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_broadcast_with_options(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as result_dir:
            ctx = mp.get_context('spawn')
            processes = []
            for rank in range(world_size):
                process = ctx.Process(
                    target=TestBroadcastOptions._hccl_broadcast_worker,
                    args=(rank, world_size, result_dir))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()
                self.assertEqual(process.exitcode, 0, "subprocess exit with abnormal code.")

            for rank in range(world_size):
                results = torch.load(
                    os.path.join(result_dir, f"rank{rank}_broadcast.pt"),
                    weights_only=True)
                self.assertTrue(torch.equal(
                    results["async_result"], torch.tensor([1.0], dtype=torch.float32)))
                self.assertTrue(torch.equal(
                    results["sync_result"], torch.tensor([0.0], dtype=torch.float32)))


if __name__ == '__main__':
    run_tests()
