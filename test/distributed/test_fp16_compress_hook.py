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
Add validation cases for torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook on NPU:
1. PyTorch community has tests in test_c10d_nccl.py, but they require NCCL backend and multi-GPU.
2. This file validates fp16_compress_hook on NPU with single card:
   - process_group=None falls back to dist.group.WORLD;
   - process_group with world_size > 1 divides the gradient by world size;
   - tuple and GradBucket bucket input forms;
   - dist.all_reduce exception propagation;
   - all_reduce is called with group and async_op=True.
"""

from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default

from torch_npu.testing.testcase import TestCase, run_tests


class _FakeGradBucket:
    def __init__(self, t):
        self._buffer = t

    def buffer(self):
        return self._buffer


class _FakeProcessGroup:
    def __init__(self, size=1):
        self._size = size

    def size(self):
        return self._size


class _FakeWork:
    def __init__(self, t):
        self._tensor = t

    def get_future(self):
        fut = torch.futures.Future()
        fut.set_result([self._tensor])
        return fut


def _fake_allreduce(tensor, group=None, async_op=False):
    return _FakeWork(tensor)


class TestFp16CompressHook(TestCase):

    def test_fp16_compress_hook_is_callable(self):
        self.assertTrue(callable(default.fp16_compress_hook))

    def test_fp16_compress_hook_dtype_conversion(self):
        """Verify fp16 compress -> allreduce -> decompress dtype behavior."""
        tensor = torch.randn(4, 4, dtype=torch.float32, device="npu")

        bucket = _FakeGradBucket(tensor.clone())
        pg = _FakeProcessGroup()
        with mock.patch.object(dist, "all_reduce", side_effect=_fake_allreduce):
            fut = default.fp16_compress_hook(pg, bucket)
            result = fut.wait()

        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, tensor.shape)
        expected = tensor.to(torch.float16).to(torch.float32)
        self.assertTrue(torch.equal(result, expected))

    def test_fp16_compress_hook_process_group_none(self):
        """process_group=None falls back to dist.group.WORLD with async_op=True."""
        if not dist.is_available():
            self.skipTest("distributed not available")

        tensor = torch.randn(4, 4, dtype=torch.float32, device="npu")
        bucket = _FakeGradBucket(tensor.clone())

        # dist.group.WORLD is only meaningful once a process group is
        # initialized, so set up a single-rank group for the None fallback.
        dist.init_process_group(
            backend="hccl",
            init_method="tcp://127.0.0.1:29501",
            rank=0,
            world_size=1,
        )
        try:
            with mock.patch.object(dist, "all_reduce", side_effect=_fake_allreduce) as mock_ar:
                fut = default.fp16_compress_hook(None, bucket)
                fut.wait()

            self.assertIs(mock_ar.call_args.kwargs["group"], dist.group.WORLD)
            self.assertIs(mock_ar.call_args.kwargs["async_op"], True)
        finally:
            dist.destroy_process_group()

    def test_fp16_compress_hook_allreduce_world_size(self):
        """With world_size > 1 the compressed tensor is divided by the world size."""
        tensor = torch.randn(4, 4, dtype=torch.float32, device="npu")
        bucket = _FakeGradBucket(tensor.clone())
        pg = _FakeProcessGroup(size=4)

        with mock.patch.object(dist, "all_reduce", side_effect=_fake_allreduce) as mock_ar:
            fut = default.fp16_compress_hook(pg, bucket)
            result = fut.wait()

        ar_tensor = mock_ar.call_args.args[0]
        self.assertIs(mock_ar.call_args.kwargs["group"], pg)
        self.assertIs(mock_ar.call_args.kwargs["async_op"], True)
        self.assertEqual(ar_tensor.dtype, torch.float16)
        self.assertTrue(torch.equal(ar_tensor, tensor.to(torch.float16) / 4))
        expected = (tensor.to(torch.float16) / 4).to(torch.float32)
        self.assertTrue(torch.equal(result, expected))

    def test_fp16_compress_hook_tuple_bucket(self):
        """The implementation also accepts a tuple of tensors as the bucket."""
        tensor = torch.randn(4, 4, dtype=torch.float32, device="npu")
        pg = _FakeProcessGroup()

        with mock.patch.object(dist, "all_reduce", side_effect=_fake_allreduce):
            fut = default.fp16_compress_hook(pg, (tensor.clone(),))
            result = fut.wait()

        expected = tensor.to(torch.float16).to(torch.float32)
        self.assertTrue(torch.equal(result, expected))

    def test_fp16_compress_hook_allreduce_exception(self):
        """An exception raised by dist.all_reduce propagates to the caller."""
        tensor = torch.randn(4, 4, dtype=torch.float32, device="npu")
        bucket = _FakeGradBucket(tensor.clone())
        pg = _FakeProcessGroup()

        with mock.patch.object(
            dist, "all_reduce", side_effect=RuntimeError("allreduce failed")
        ):
            with self.assertRaises(RuntimeError):
                default.fp16_compress_hook(pg, bucket)


if __name__ == "__main__":
    run_tests()
