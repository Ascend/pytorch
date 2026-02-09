import os
import unittest
import subprocess
import sys
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase
from torch_npu.testing.testcase import run_tests


class TestTorchNpuLogs(TestCase):
    def setUp(self):
        self.original_torch_npu_logs = os.getenv('TORCH_NPU_LOGS')
        self.original_torch_npu_logs_filter = os.getenv('TORCH_NPU_LOGS_FILTER')

    def tearDown(self):
        if self.original_torch_npu_logs is not None:
            os.environ['TORCH_NPU_LOGS'] = self.original_torch_npu_logs
        else:
            del os.environ['TORCH_NPU_LOGS']
            
        if self.original_torch_npu_logs_filter is not None:
            os.environ['TORCH_NPU_LOGS_FILTER'] = self.original_torch_npu_logs_filter
        else:
            del os.environ['TORCH_NPU_LOGS_FILTER']

    def _run_test(self, logs, logs_filter):
        os.environ['TORCH_NPU_LOGS'] = logs
        os.environ['TORCH_NPU_LOGS_FILTER'] = logs_filter

        result = subprocess.run(
            [sys.executable, '-c', self._get_test_code()],
            capture_output=True,
            text=True,
            env=os.environ
        )

        return result.stdout + result.stderr

    def _get_test_code(self):
        return """

import torch
import torch_npu

matrix1 = torch.randn(3, 3).npu()
matrix2 = torch.randn(3, 3).npu()

result_matmul = torch.matmul(matrix1, matrix2)

result_add = torch.add(matrix1, matrix2)
"""

    def test_no_filter(self):
        # 不设置过滤器，所有日志输出
        output = self._run_test('op_plugin', '')
        self.assertIn('mm', output)
        self.assertIn('add', output)

    def test_white_list(self):
        # 白名单过滤，只输出 matmul
        output = self._run_test('op_plugin', '+mm')
        self.assertIn('mm', output)
        self.assertNotIn('add', output)

    def test_black_list(self):
        # 黑名单过滤，排除 add
        output = self._run_test('op_plugin', '-add')
        self.assertIn('mm', output)
        self.assertNotIn('add', output)

if __name__ == '__main__':
    run_tests()
