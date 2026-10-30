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
Add validation cases for torch.distributed.tensor.experimental APIs on NPU
(PyTorch v2.7.1 variant; keep in sync with the other version variants):
1. PyTorch community tests exercise context_parallel() mainly through
   end-to-end SDPA scenarios and lack direct validations of the context
   manager contract: argument validation, in-place buffer shard/restore
   semantics, and load-balance option behaviors.
2. This file validates
   torch.distributed.tensor.experimental.context_parallel (extendable).
"""

import torch
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import _cp_options
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase

import torch_npu.distributed.tensor.experimental._context_parallel._attention  # noqa: F401
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU, with_comms


class TestContextParallelAPI(DTensorTestBase):

    def tearDown(self) -> None:
        # Tests set the global load-balance option directly; reset it to the
        # baseline after each test so the value never leaks across tests.
        _cp_options.enable_load_balance = False
        super().tearDown()

    @property
    def world_size(self) -> int:
        device_count = torch.npu.device_count() if torch.npu.is_available() else 0
        return min(8, device_count) if device_count >= 2 else 2

    def _build_mesh(self) -> DeviceMesh:
        return DeviceMesh(self.device_type, torch.arange(0, self.world_size))

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_shard_and_restore_sequential(self) -> None:
        # Buffers are sharded in-place with torch.chunk semantics and restored
        # to the original content after the context exits.
        _cp_options.enable_load_balance = False
        mesh = self._build_mesh()
        ws = self.world_size
        rank = int(self.rank)
        original = torch.arange(4 * ws, dtype=torch.float32, device=self.device_type)
        buffer = original.clone()
        with context_parallel(mesh, buffers=[buffer], buffer_seq_dims=[0]):
            self.assertEqual(buffer.shape, (4,))
            self.assertEqual(buffer, original.chunk(ws, dim=0)[rank])
        self.assertEqual(buffer.shape, (4 * ws,))
        self.assertEqual(buffer, original)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_shard_multiple_seq_dims(self) -> None:
        # Each buffer is sharded along its own sequence dimension; tuple
        # containers and an explicitly empty no_restore_buffers are accepted.
        _cp_options.enable_load_balance = False
        mesh = self._build_mesh()
        ws = self.world_size
        rank = int(self.rank)
        buf0 = torch.arange(6 * ws, dtype=torch.float32, device=self.device_type).reshape(2 * ws, 3)
        buf1 = torch.arange(4 * ws, dtype=torch.float32, device=self.device_type).reshape(2, 2 * ws)
        buf2 = torch.arange(4 * ws, dtype=torch.float32, device=self.device_type).reshape(2, 2, ws)
        expected0, expected1, expected2 = buf0.clone(), buf1.clone(), buf2.clone()
        with context_parallel(
            mesh,
            buffers=(buf0, buf1, buf2),
            buffer_seq_dims=(0, 1, 2),
            no_restore_buffers=set(),
        ):
            self.assertEqual(buf0.shape, (2, 3))
            self.assertEqual(buf0, expected0.chunk(ws, dim=0)[rank])
            self.assertEqual(buf1.shape, (2, 2))
            self.assertEqual(buf1, expected1.chunk(ws, dim=1)[rank])
            self.assertEqual(buf2.shape, (2, 2, 1))
            self.assertEqual(buf2, expected2.chunk(ws, dim=2)[rank])
        self.assertEqual(buf0, expected0)
        self.assertEqual(buf1, expected1)
        self.assertEqual(buf2, expected2)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_no_restore_buffers(self) -> None:
        # Buffers listed in no_restore_buffers keep their shards after the
        # context exits, while the others are restored.
        _cp_options.enable_load_balance = False
        mesh = self._build_mesh()
        ws = self.world_size
        rank = int(self.rank)
        restored = torch.arange(2 * ws, dtype=torch.float32, device=self.device_type)
        skipped = torch.arange(2 * ws, dtype=torch.float32, device=self.device_type)
        expected = restored.clone()
        with context_parallel(
            mesh,
            buffers=[restored, skipped],
            buffer_seq_dims=[0, 0],
            no_restore_buffers={skipped},
        ):
            self.assertEqual(restored.shape, (2,))
            self.assertEqual(skipped.shape, (2,))
        self.assertEqual(restored.shape, (2 * ws,))
        self.assertEqual(restored, expected)
        self.assertEqual(skipped.shape, (2,))
        self.assertEqual(skipped, expected.chunk(ws, dim=0)[rank])

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_head_tail_load_balance(self) -> None:
        # With load balancing enabled and a sequence length divisible by
        # 2 * world_size, each rank keeps the head-tail rearranged shard
        # (PyTorch v2.7.1 implements this via _RoundRobinLoadBalancer).
        _cp_options.enable_load_balance = True
        mesh = self._build_mesh()
        ws = self.world_size
        rank = int(self.rank)
        seq_len = 2 * ws
        original = torch.arange(seq_len, dtype=torch.float32, device=self.device_type)
        buffer = original.clone()
        with context_parallel(mesh, buffers=[buffer], buffer_seq_dims=[0]):
            self.assertTrue(_cp_options.enable_load_balance)
            self.assertEqual(buffer.shape, (seq_len // ws,))
            expected = torch.cat(
                (original[rank : rank + 1], original[seq_len - rank - 1 : seq_len - rank])
            )
            self.assertEqual(buffer, expected)
        self.assertEqual(buffer, original)
        self.assertTrue(_cp_options.enable_load_balance)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_load_balance_non_divisible_seq_len_raises(self) -> None:
        # PyTorch v2.7.1 always picks _RoundRobinLoadBalancer when
        # enable_load_balance is set, and sequence lengths not divisible by
        # 2 * world_size fail its divisibility assert; the graceful fallback
        # to regular chunks only exists on torch main (pytorch#183968). The
        # assert fires before any buffer mutation.
        _cp_options.enable_load_balance = True
        mesh = self._build_mesh()
        ws = self.world_size
        original = torch.arange(3 * ws, dtype=torch.float32, device=self.device_type)
        buffer = original.clone()
        with self.assertRaises(AssertionError):
            with context_parallel(mesh, buffers=[buffer], buffer_seq_dims=[0]):
                pass
        self.assertEqual(buffer, original)

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_mismatched_buffer_seq_dims_raises(self) -> None:
        # buffer_seq_dims must have the same number of elements as buffers.
        mesh = self._build_mesh()
        buffer = torch.arange(8, dtype=torch.float32, device=self.device_type)
        with self.assertRaises(ValueError):
            with context_parallel(mesh, buffers=[buffer], buffer_seq_dims=[]):
                pass
        with self.assertRaises(ValueError):
            with context_parallel(
                mesh, buffers=[buffer, buffer.clone()], buffer_seq_dims=[0]
            ):
                pass

    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_no_restore_not_subset_raises(self) -> None:
        # no_restore_buffers must be a subset of buffers.
        mesh = self._build_mesh()
        buffer = torch.arange(8, dtype=torch.float32, device=self.device_type)
        outsider = torch.arange(4, dtype=torch.float32, device=self.device_type)
        with self.assertRaises(ValueError):
            with context_parallel(
                mesh,
                buffers=[buffer],
                buffer_seq_dims=[0],
                no_restore_buffers={outsider},
            ):
                pass


if __name__ == "__main__":
    run_tests()
