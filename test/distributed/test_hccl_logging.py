import os
import subprocess
import sys
import unittest

import torch
import torch.distributed as dist
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


def run_all_reduce():
    torch_npu.npu.set_device(0)
    dist.init_process_group("hccl", store=dist.HashStore(), rank=0, world_size=1)
    try:
        dist.all_reduce(torch.ones(1, device="npu"))
        torch_npu.npu.synchronize()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(torch_npu.npu.is_available(), "requires an NPU")
class TestHcclLogging(TestCase):
    def test_torch_npu_logs_levels(self):
        for setting, info, debug in ((None, False, False), ("hccl", True, False), ("+hccl", True, True)):
            with self.subTest(setting=setting):
                env = os.environ.copy()
                if setting is None:
                    env.pop("TORCH_NPU_LOGS", None)
                else:
                    env["TORCH_NPU_LOGS"] = setting
                result = subprocess.run(
                    [sys.executable, "-c", "from test_hccl_logging import run_all_reduce; run_all_reduce()"],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    env=env, capture_output=True, text=True, timeout=60,
                )
                stderr = result.stderr
                self.assertEqual(result.returncode, 0, stderr)
                self.assertEqual("[SET] Publishing HCCL RootInfo" in stderr, info, stderr)
                self.assertEqual("Calling hcclAllReduce" in stderr, debug, stderr)


if __name__ == "__main__":
    run_tests()
