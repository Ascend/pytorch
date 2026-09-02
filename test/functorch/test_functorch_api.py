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
Add validation cases for torch._C._functorch APIs on NPU:
1. PyTorch community lacks direct test cases for the following APIs,
   so this file is added.
2. This file validates torch._C._functorch.is_batchedtensor (extendable).
"""

import torch
from torch._C._functorch import (
    _add_batch_dim,
    _vmap_decrement_nesting,
    _vmap_increment_nesting,
    get_unwrapped,
    is_batchedtensor,
)
from torch.testing._internal.common_utils import run_tests, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestFunctorchIsBatchedTensor(TestCase):

    def test_is_batchedtensor_plain_tensor(self):
        # a normal tensor is not a BatchedTensor
        x = torch.randn(2, 3).to(device_type)
        self.assertFalse(is_batchedtensor(x))

    def test_is_batchedtensor_inside_vmap(self):
        # the tensor passed into a vmap-ed function is a BatchedTensor
        seen = []

        def fn(t):
            seen.append(is_batchedtensor(t))
            return t.sum()

        x = torch.randn(4, 3).to(device_type)
        torch.vmap(fn)(x)
        self.assertEqual(seen, [True])

    def test_is_batchedtensor_manual_batch_dim(self):
        # manually wrapped BatchedTensor is recognized, unwrapping restores False
        x = torch.randn(3, 5).to(device_type)
        level = _vmap_increment_nesting(3, "error")
        try:
            batched = _add_batch_dim(x, 0, level)
            self.assertTrue(is_batchedtensor(batched))
            self.assertFalse(is_batchedtensor(get_unwrapped(batched)))
        finally:
            _vmap_decrement_nesting()
        # The finally block must restore the vmap nesting level so later tests
        # are not polluted. Re-incrementing yields the same level we started
        # from, proving the nesting counter was fully restored.
        level_after = _vmap_increment_nesting(3, "error")
        try:
            self.assertEqual(level_after, level)
        finally:
            _vmap_decrement_nesting()

    def test_is_batchedtensor_nested_vmap(self):
        # BatchedTensor of nested vmap is still a BatchedTensor
        seen = []

        def fn(t):
            seen.append(is_batchedtensor(t))
            return t.sum()

        x = torch.randn(2, 3, 4).to(device_type)
        torch.vmap(torch.vmap(fn))(x)
        self.assertEqual(seen, [True])

    def test_is_batchedtensor_outside_vmap(self):
        # the tensor is no longer batched after vmap returns
        x = torch.randn(4, 3).to(device_type)
        out = torch.vmap(lambda t: t * 2)(x)
        self.assertFalse(is_batchedtensor(out))

    def test_is_batchedtensor_various_dtypes(self):
        # the result only depends on batching, not on dtype
        for dtype in (torch.float32, torch.float16, torch.int32, torch.bool):
            x = torch.ones(2, 3, dtype=dtype).to(device_type)
            self.assertFalse(is_batchedtensor(x))

    def test_is_batchedtensor_non_tensor_input(self):
        # non-tensor input is rejected
        with self.assertRaises(TypeError):
            is_batchedtensor(1)

    def test_is_batchedtensor_zero_dim_tensor(self):
        # a 0-d (scalar) tensor is not a BatchedTensor
        x = torch.tensor(7).to(device_type)
        self.assertFalse(is_batchedtensor(x))

    def test_is_batchedtensor_illegal_inputs(self):
        # None / str / arbitrary object inputs are rejected
        for bad in (None, "x", object()):
            with self.assertRaises(TypeError):
                is_batchedtensor(bad)

    def test_is_batchedtensor_other_device(self):
        # device of the tensor does not affect the batching check
        x_npu = torch.randn(2, 3).to(device_type)
        x_cpu = torch.randn(2, 3)
        self.assertFalse(is_batchedtensor(x_npu))
        self.assertFalse(is_batchedtensor(x_cpu))

    def test_is_batchedtensor_nesting_balance(self):
        # A full increment/decrement cycle must restore the vmap nesting level,
        # guarding against corrupted nesting state.
        level = _vmap_increment_nesting(2, "error")
        try:
            self.assertGreaterEqual(level, 1)
        finally:
            _vmap_decrement_nesting()


if __name__ == "__main__":
    run_tests()
