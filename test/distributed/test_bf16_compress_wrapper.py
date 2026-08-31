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
Add validation cases for torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper on NPU:

1. PyTorch community has tests in test_c10d_nccl.py, but they require NCCL backend and multi-GPU.
2. This file validates bf16_compress_wrapper functionality on NPU with single card.

The wrapper should:
- Pass a bfloat16 tensor to the underlying communication hook.
- Return the bfloat16 result; the original dtype is restored when the result is
  copied back into the fp32 gradient buffer (as downstream consumers do).
"""

import torch
import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default

from torch_npu.testing.testcase import TestCase, run_tests


class FakeGradBucket:
    """Minimal fake of a GradientBucket exposing buffer()/set_buffer()."""

    def __init__(self, tensor):
        self._buffer = tensor
        self.set_buffer_called = False

    def buffer(self):
        return self._buffer

    def set_buffer(self, tensor):
        self.set_buffer_called = True
        self._buffer = tensor


class TestBf16CompressWrapper(TestCase):

    def test_bf16_compress_wrapper_returns_callable(self):
        def dummy_hook(state, bucket):
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        wrapped = default.bf16_compress_wrapper(dummy_hook)

        self.assertTrue(callable(wrapped))

    def test_bf16_compress_wrapper_dtype_conversion(self):
        original_tensor = torch.randn(
            4,
            4,
            dtype=torch.float32,
            device="npu",
        )

        def check_hook(state, bucket):
            compressed = bucket.buffer()

            # Compression: the underlying hook receives a bf16 tensor.
            self.assertEqual(
                compressed.dtype,
                torch.bfloat16,
            )
            self.assertTrue(
                torch.equal(
                    compressed,
                    original_tensor.to(torch.bfloat16),
                )
            )

            fut = torch.futures.Future()
            fut.set_result(compressed)
            return fut

        bucket = FakeGradBucket(original_tensor.clone())

        wrapped = default.bf16_compress_wrapper(check_hook)

        fut = wrapped(None, bucket)
        result = fut.wait()

        # Decompression: the future resolves to the hook output (bf16),
        # which the wrapper copies back into the bucket buffer.
        self.assertEqual(
            result.dtype,
            torch.bfloat16,
        )

        self.assertEqual(
            result.shape,
            original_tensor.shape,
        )

        self.assertTrue(
            torch.equal(
                result,
                original_tensor.to(torch.bfloat16),
            )
        )

        # Original dtype restore: even though the wrapper returns bf16, copying
        # the result into an fp32 tensor restores float32.
        restored = original_tensor.clone()
        restored.copy_(result)
        self.assertEqual(
            restored.dtype,
            torch.float32,
        )
        self.assertTrue(
            torch.equal(
                restored,
                original_tensor.to(torch.bfloat16).to(torch.float32),
            )
        )

    def test_bf16_compress_wrapper_allreduce(self):
        if not dist.is_available():
            self.skipTest("distributed not available")

        # Single-rank process group to exercise a real allreduce collective on NPU.
        dist.init_process_group(
            backend="hccl",
            init_method="tcp://127.0.0.1:29501",
            rank=0,
            world_size=1,
        )
        try:
            process_group = dist.group.WORLD
            original_tensor = torch.randn(
                4,
                4,
                dtype=torch.float32,
                device="npu",
            )
            bucket = FakeGradBucket(original_tensor.clone())

            # bf16_compress_wrapper casts the bucket to bf16, then the wrapped
            # allreduce_hook runs a real dist.all_reduce before decompression.
            wrapped = default.bf16_compress_wrapper(default.allreduce_hook)

            fut = wrapped(process_group, bucket)
            result = fut.wait()

            # Decompression: the wrapper returns the bucket buffer (bf16) that
            # holds the allreduced result.
            self.assertEqual(
                result.dtype,
                torch.bfloat16,
            )

            self.assertEqual(
                result.shape,
                original_tensor.shape,
            )

            # With world_size=1 the allreduce is an identity; only the bf16
            # rounding applies to the result.
            self.assertTrue(
                torch.equal(
                    result,
                    original_tensor.to(torch.bfloat16),
                )
            )
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
