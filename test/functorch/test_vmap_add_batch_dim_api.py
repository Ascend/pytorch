# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
Add validation cases for torch._functorch.vmap._add_batch_dim API on NPU:
1. PyTorch community lacks sufficient and direct API validations for this API, so this file is added.
2. This file validates torch._functorch.vmap._add_batch_dim (extendable).
"""
import torch
from torch._functorch.vmap import _add_batch_dim
from torch.testing._internal.common_utils import run_tests, TestCase

device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
)


class TestVmapAddBatchDim(TestCase):
    def test_add_batch_dim_basic(self):
        """Test basic _add_batch_dim functionality on NPU"""
        x = torch.randn(3, 4).to(device_type)
        vmap_level = 0
        batched = _add_batch_dim(x, 0, vmap_level)
        self.assertIsNotNone(batched)
        self.assertIsInstance(batched, torch.Tensor)
        self.assertEqual(batched.device.type, device_type)
        self.assertEqual(batched.shape, (4,))

    def test_add_batch_dim_with_vmap(self):
        """Test _add_batch_dim works correctly with vmap on NPU"""
        x = torch.randn(2, 3).to(device_type)
        y = torch.randn(2, 3).to(device_type)

        def dot_row(a, b):
            return (a * b).sum(dim=-1)

        result = torch.vmap(dot_row)(x, y)
        expected = dot_row(x, y)
        self.assertEqual(result, expected)
        self.assertEqual(result.device.type, device_type)

    def test_add_batch_dim_nested_vmap(self):
        """Test nested vmap with _add_batch_dim on NPU"""
        x = torch.randn(2, 3, 4).to(device_type)
        y = torch.randn(2, 3, 4).to(device_type)

        def matmul_row(a, b):
            return (a * b).sum(dim=-1)

        result = torch.vmap(torch.vmap(matmul_row))(x, y)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.device.type, device_type)

    def test_add_batch_dim_with_model(self):
        """Test _add_batch_dim with a simple model on NPU"""
        model = torch.nn.Linear(4, 2).to(device_type)
        x = torch.randn(3, 4).to(device_type)

        result = torch.vmap(lambda x: model(x))(x)
        expected = model(x)
        self.assertEqual(result, expected)
        self.assertEqual(result.device.type, device_type)

    def test_add_batch_dim_in_dims(self):
        """Test _add_batch_dim with different in_dims on NPU"""
        x = torch.randn(3, 4, 5).to(device_type)

        def identity(x):
            return x

        # Test in_dims=0 (default)
        result0 = torch.vmap(identity, in_dims=0)(x)
        self.assertEqual(result0.shape, (3, 4, 5))

        # Test in_dims=1
        result1 = torch.vmap(identity, in_dims=1)(x)
        self.assertEqual(result1.shape, (4, 3, 5))

        # Test in_dims=-1
        result_neg1 = torch.vmap(identity, in_dims=-1)(x)
        self.assertEqual(result_neg1.shape, (5, 3, 4))

    def test_add_batch_dim_out_dims(self):
        """Test _add_batch_dim with different out_dims on NPU"""
        x = torch.randn(3, 4).to(device_type)

        def identity(x):
            return x

        # Test out_dims=0 (default)
        result0 = torch.vmap(identity, out_dims=0)(x)
        self.assertEqual(result0.shape, (3, 4))

        # Test out_dims=1
        result1 = torch.vmap(identity, out_dims=1)(x)
        self.assertEqual(result1.shape, (4, 3))

    def test_add_batch_dim_with_grad(self):
        """Test _add_batch_dim works with gradient computation on NPU"""
        x = torch.randn(3, 3, device=device_type, requires_grad=True)
        w = torch.randn(3, 3, device=device_type, requires_grad=True)

        def fn(x, w):
            return (x * w).sum(dim=-1)

        result = torch.vmap(fn)(x, w)
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        self.assertEqual(x.grad.device.type, device_type)
        self.assertEqual(w.grad.device.type, device_type)

    def test_add_batch_dim_direct_3d_batch_dim_0(self):
        """Direct API call: 3D tensor with batch_dim=0 returns shape=(4,5) (inner value sliced at dim 0)."""
        x = torch.randn(3, 4, 5, dtype=torch.float32).to(device_type)
        vmap_level = 0
        batched = _add_batch_dim(x, 0, vmap_level)
        self.assertIsNotNone(batched)
        self.assertEqual(batched.shape, (4, 5))
        self.assertEqual(batched.device.type, device_type)
        self.assertEqual(batched.dtype, torch.float32)

    def test_add_batch_dim_direct_3d_batch_dim_1(self):
        """Direct API call: 3D tensor with batch_dim=1 returns shape=(3,5)."""
        x = torch.randn(3, 4, 5, dtype=torch.float32).to(device_type)
        vmap_level = 0
        batched = _add_batch_dim(x, 1, vmap_level)
        self.assertIsNotNone(batched)
        self.assertEqual(batched.shape, (3, 5))
        self.assertEqual(batched.device.type, device_type)

    def test_add_batch_dim_direct_3d_batch_dim_2(self):
        """Direct API call: 3D tensor with batch_dim=2 returns shape=(3,4)."""
        x = torch.randn(3, 4, 5, dtype=torch.float32).to(device_type)
        vmap_level = 0
        batched = _add_batch_dim(x, 2, vmap_level)
        self.assertIsNotNone(batched)
        self.assertEqual(batched.shape, (3, 4))
        self.assertEqual(batched.device.type, device_type)

    def test_add_batch_dim_direct_preserves_dtype_and_device(self):
        """Direct API call preserves dtype and device."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).to(device_type)
        vmap_level = 0
        batched = _add_batch_dim(x, 0, vmap_level)
        self.assertEqual(batched.dtype, torch.float32)
        self.assertEqual(batched.device.type, device_type)
        self.assertEqual(batched.shape, (2,))

    def test_add_batch_dim_invalid_batch_dim(self):
        """Direct API call with out-of-range batch_dim wraps (batch_dim % x.ndim)."""
        x = torch.randn(3, 4).to(device_type)
        # batch_dim=5 on 2D tensor acts as batch_dim=1 (5 % 2 ≡ 1)
        batched = _add_batch_dim(x, 5, 0)
        self.assertEqual(batched.shape, (3,))
        self.assertEqual(batched.device.type, device_type)

    def test_add_batch_dim_multiple_levels(self):
        """Test _add_batch_dim / _remove_batch_dim roundtrip restores original tensor."""
        from torch._functorch.vmap import _remove_batch_dim
        x = torch.randn(3, 4).to(device_type)
        b = _add_batch_dim(x, 0, 0)
        restored = _remove_batch_dim(b, 0, x.shape[0], 0)
        self.assertEqual(restored.shape, x.shape)
        self.assertEqual(restored.device.type, device_type)
        self.assertTrue(torch.equal(restored.cpu(), x.cpu()))


if __name__ == "__main__":
    run_tests()
