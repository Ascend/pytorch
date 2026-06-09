# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.get_worker_info 接口功能正确性
API 名称：torch.utils.data.get_worker_info
API 签名：torch.utils.data.get_worker_info() -> WorkerInfo or None

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 主进程调用       | 返回 None                           | 已覆盖                  |

未覆盖项及原因：
- worker 内调用需 DataLoader(num_workers>0) 环境

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.utils.data import get_worker_info

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase
    def run_tests():
        unittest.main(argv=sys.argv)


class TestUtilsDataGetWorkerInfo(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_main_process_returns_none(self):
        """Verify get_worker_info returns None in main process."""
        result = get_worker_info()
        self.assertIsNone(result)


if __name__ == "__main__":
    run_tests()
