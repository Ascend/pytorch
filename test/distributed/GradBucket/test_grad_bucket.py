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
Add validation cases for torch.distributed.GradBucket APIs on NPU:
1. PyTorch community tests exercise GradBucket.buffer() only indirectly via
   CUDA-bound DDP comm-hook scenarios and lack direct contract checks on NPU.
2. This file validates GradBucket.buffer/set_buffer together with the sibling
   accessors index/is_last/gradients/parameters (extendable).
3. Complex-bucket real-view layout and sparse-gradient buckets are
   version-divergent and left to upstream cases; multi-rank DDP hook parity
   on NPU is covered by test/distributed/test_comm_hook.py.
"""

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class _TwoLinearModule(nn.Module):
    # Two linear layers whose parameters are bucketed by the reducer.
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=True)
        self.fc2 = nn.Linear(3, 2, bias=True)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TestGradBucketAPI(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Single-process group: HCCL on NPU exercises the real device comm
        # path, gloo keeps the CPU fallback runnable.
        handle, cls.store_path = tempfile.mkstemp()
        os.close(handle)
        backend = "hccl" if device_type == "npu" else "gloo"
        dist.init_process_group(
            backend, store=dist.FileStore(cls.store_path, 1), rank=0, world_size=1
        )

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        # FileStore may remove the backing file on process group teardown.
        if os.path.exists(cls.store_path):
            os.remove(cls.store_path)
        super().tearDownClass()

    def _build_model(self, dtype):
        # Deterministic weights so reference and reducer runs match exactly.
        model = _TwoLinearModule()
        with torch.no_grad():
            for index, param in enumerate(model.parameters()):
                param.fill_(0.1 * (index + 1))
        return model.to(device=device_type, dtype=dtype)

    def _input(self, dtype):
        return torch.arange(8, dtype=torch.float32).reshape(2, 4).to(
            device=device_type, dtype=dtype
        )

    def _expected_grads(self, dtype):
        reference = self._build_model(dtype)
        loss = reference(self._input(dtype)).sum()
        loss.backward()
        return [param.grad.detach().clone() for param in reference.parameters()]

    def _make_reducer(self, params, bucket_indices, gradient_as_bucket_view=False):
        return dist.Reducer(
            params,
            bucket_indices,
            [dist._DEFAULT_FIRST_BUCKET_BYTES for _ in bucket_indices],
            dist.group.WORLD,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

    def _run_backward(self, model, reducer, x):
        reducer.prepare_for_forward()
        loss = model(x).sum()
        reducer.prepare_for_backward(loss)
        loss.backward()

    @staticmethod
    def _completed_future(tensor):
        fut = torch.futures.Future()
        fut.set_result(tensor)
        return fut

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_buffer_contract_single_bucket(self):
        # buffer() returns the flattened 1D bucket buffer: numel/dtype/device
        # follow the bucket parameters, gradients are laid out in parameter
        # order, and gradients()/parameters() expose views of the same bucket.
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                expected = self._expected_grads(dtype)
                model = self._build_model(dtype)
                params = list(model.parameters())
                reducer = self._make_reducer(params, [list(range(len(params)))])

                def hook(state, bucket):
                    buffer = bucket.buffer()
                    grads = bucket.gradients()
                    self.assertEqual(buffer.dim(), 1)
                    self.assertEqual(buffer.numel(), sum(p.numel() for p in params))
                    self.assertEqual(buffer.dtype, dtype)
                    self.assertEqual(buffer.device.type, device_type)
                    self.assertEqual(buffer.data_ptr(), grads[0].data_ptr())
                    offset = 0
                    for grad, expect in zip(grads, expected):
                        self.assertEqual(grad, expect)
                        self.assertEqual(
                            buffer[offset : offset + expect.numel()],
                            expect.reshape(-1),
                        )
                        offset += expect.numel()
                    self.assertEqual(
                        [p.data_ptr() for p in bucket.parameters()],
                        [p.data_ptr() for p in params],
                    )
                    self.assertEqual(bucket.index(), 0)
                    self.assertTrue(bucket.is_last())
                    return self._completed_future(buffer)

                dist._register_comm_hook(reducer, None, hook)
                self._run_backward(model, reducer, self._input(dtype))
                for param, expect in zip(params, expected):
                    self.assertEqual(param.grad, expect)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_buffer_contract_multiple_buckets(self):
        # One bucket per parameter: each buffer() maps to a single parameter
        # and index()/is_last() reflect the bucket position.
        expected = self._expected_grads(torch.float32)
        model = self._build_model(torch.float32)
        params = list(model.parameters())
        bucket_indices = [[index] for index in range(len(params))]
        reducer = self._make_reducer(params, bucket_indices)
        records = []

        def hook(records, bucket):
            index = bucket.index()
            buffer = bucket.buffer()
            param = params[index]
            self.assertEqual(buffer.dim(), 1)
            self.assertEqual(buffer.numel(), param.numel())
            self.assertEqual(buffer.dtype, param.dtype)
            self.assertEqual(buffer.device.type, device_type)
            self.assertEqual(buffer, expected[index].reshape(-1))
            self.assertEqual(bucket.is_last(), index == len(bucket_indices) - 1)
            records.append(index)
            return self._completed_future(buffer)

        dist._register_comm_hook(reducer, records, hook)
        self._run_backward(model, reducer, self._input(torch.float32))
        self.assertEqual(sorted(records), list(range(len(params))))
        for param, expect in zip(params, expected):
            self.assertEqual(param.grad, expect)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_buffer_inplace_mutation(self):
        # buffer() shares storage with the bucket: in-place updates through
        # the returned tensor are visible in gradients() and in final grads.
        expected = self._expected_grads(torch.float32)
        model = self._build_model(torch.float32)
        params = list(model.parameters())
        reducer = self._make_reducer(params, [list(range(len(params)))])

        def hook(state, bucket):
            bucket.buffer().add_(1.0)
            for grad, expect in zip(bucket.gradients(), expected):
                self.assertEqual(grad, expect + 1)
            return self._completed_future(bucket.buffer())

        dist._register_comm_hook(reducer, None, hook)
        self._run_backward(model, reducer, self._input(torch.float32))
        for param, expect in zip(params, expected):
            self.assertEqual(param.grad, expect + 1)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_set_buffer_replacement(self):
        # set_buffer() replaces the bucket buffer, buffer() then returns the
        # new tensor, and the hook result is written back to parameter grads.
        expected = self._expected_grads(torch.float32)
        model = self._build_model(torch.float32)
        params = list(model.parameters())
        reducer = self._make_reducer(params, [list(range(len(params)))])

        def hook(state, bucket):
            scaled = bucket.buffer() * 2
            with self.assertRaises(TypeError):
                bucket.set_buffer(1)
            bucket.set_buffer(scaled)
            self.assertEqual(bucket.buffer().data_ptr(), scaled.data_ptr())
            return self._completed_future(bucket.buffer())

        dist._register_comm_hook(reducer, None, hook)
        self._run_backward(model, reducer, self._input(torch.float32))
        for param, expect in zip(params, expected):
            self.assertEqual(param.grad, expect * 2)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_buffer_gradient_as_bucket_view(self):
        # With gradient_as_bucket_view=True the parameter grads are views into
        # the bucket buffer; with False they are copies written back from it.
        for gradient_as_bucket_view in (False, True):
            with self.subTest(gradient_as_bucket_view=gradient_as_bucket_view):
                expected = self._expected_grads(torch.float32)
                model = self._build_model(torch.float32)
                params = list(model.parameters())
                reducer = self._make_reducer(
                    params, [list(range(len(params)))], gradient_as_bucket_view
                )
                state = {}

                def hook(state, bucket):
                    state["buffer_ptr"] = bucket.buffer().data_ptr()
                    return self._completed_future(bucket.buffer())

                dist._register_comm_hook(reducer, state, hook)
                self._run_backward(model, reducer, self._input(torch.float32))
                for param, expect in zip(params, expected):
                    self.assertEqual(param.grad, expect)
                if gradient_as_bucket_view:
                    self.assertEqual(params[0].grad.data_ptr(), state["buffer_ptr"])
                else:
                    self.assertNotEqual(params[0].grad.data_ptr(), state["buffer_ptr"])

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_buffer_allreduce_comm_hook(self):
        # Canonical allreduce comm hook (default_hooks pattern): buffer()
        # feeds the collective on the device backend (HCCL on NPU) and the
        # result is written back to the parameter grads.
        expected = self._expected_grads(torch.float32)
        model = self._build_model(torch.float32)
        params = list(model.parameters())
        reducer = self._make_reducer(params, [list(range(len(params)))])

        def allreduce_hook(process_group, bucket):
            buffer = bucket.buffer()
            buffer.div_(process_group.size())
            return (
                dist.all_reduce(buffer, group=process_group, async_op=True)
                .get_future()
                .then(lambda fut: fut.value()[0])
            )

        dist._register_comm_hook(reducer, dist.group.WORLD, allreduce_hook)
        self._run_backward(model, reducer, self._input(torch.float32))
        for param, expect in zip(params, expected):
            self.assertEqual(param.grad, expect)

    @unittest.skipUnless(torch.npu.is_available(), "requires npu")
    def test_mixed_dtype_bucket_rejected(self):
        # A bucket must stay dtype homogeneous; the Reducer rejects mixed
        # buckets at construction, keeping the buffer() contract well-defined.
        # float16 is used because double is not supported on NPU.
        model = _TwoLinearModule().to(device=device_type)
        model.fc1.half()
        params = list(model.parameters())
        with self.assertRaisesRegex(
            RuntimeError, "All parameters in a bucket must have the same dtype"
        ):
            dist.Reducer(
                params,
                [list(range(len(params)))],
                [dist._DEFAULT_FIRST_BUCKET_BYTES],
                dist.group.WORLD,
            )


if __name__ == "__main__":
    run_tests()
