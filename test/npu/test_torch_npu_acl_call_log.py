import os
import subprocess
import sys
import unittest
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

x = torch.randn(16, device="npu")

event = torch.npu.Event(enable_timing=True)

event.record()

torch.npu.synchronize()

"""


    def test_white_list(self):
        output = self._run_test('acl', '+aclrtCreateEventWithFlag')
        self.assertIn('aclrtCreateEventWithFlag', output)

    def test_black_list(self):
        output = self._run_test('acl', '-aclrtCreateEventWithFlag')
        self.assertNotIn('aclrtCreateEventWithFlag', output)

if __name__ == '__main__':
    run_tests()
