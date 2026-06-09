# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.DataLoader 接口功能正确性
API 名称：torch.utils.data.DataLoader
API 签名：torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, ...)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 创建 DataLoader 不报错              | 已覆盖                  |
| 迭代             | 迭代返回 batch                      | 已覆盖                  |
| batch_size       | batch 维度正确                      | 已覆盖                  |
| shuffle          | shuffle=True/False                  | 已覆盖                  |
| num_workers=0    | 单进程加载                          | 已覆盖                  |

未覆盖项及原因：
- num_workers>0：多进程环境复杂
- pin_memory/collate_fn 等高级参数

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.utils.data import Dataset, DataLoader

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class _TensorDataset(Dataset):
    def __init__(self, size=10):
        self.data = torch.randn(size, 4)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestUtilsDataDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_create_dataloader(self):
        """Verify DataLoader can be created with a dataset and batch_size."""
        ds = _TensorDataset()
        loader = DataLoader(ds, batch_size=2)
        self.assertIsInstance(loader, DataLoader)

    def test_iter_returns_batches(self):
        """Verify iteration yields correct number of batches (ceil division)."""
        ds = _TensorDataset(10)
        loader = DataLoader(ds, batch_size=3, num_workers=0)
        batches = list(loader)
        self.assertEqual(len(batches), 4)  # ceil(10/3) = 4

    def test_batch_shape(self):
        """Verify first batch has expected shape."""
        ds = _TensorDataset(10)
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        first_batch = next(iter(loader))
        self.assertEqual(first_batch.shape[0], 4)
        self.assertEqual(first_batch.shape[1], 4)

    def test_shuffle_true(self):
        """Verify DataLoader with shuffle=True produces correct shape."""
        ds = _TensorDataset(10)
        loader = DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        self.assertEqual(batch.shape, torch.Size([10, 4]))

    def test_npu_tensor_dataset(self):
        """Verify DataLoader works with NPU tensor-returning dataset."""
        class _NpuDataset(Dataset):
            def __len__(self):
                return 8
            def __getitem__(self, idx):
                return torch.tensor(idx, device='npu')
        ds = _NpuDataset()
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        self.assertEqual(batch.shape, torch.Size([4]))


if __name__ == "__main__":
    run_tests()
