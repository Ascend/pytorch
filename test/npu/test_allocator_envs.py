import os
import math
import subprocess
import sys
import textwrap
import torch
import torch_npu

import torch_npu.npu.utils as utils

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAllocator(TestCase):
    def test_huge_memory_alloc_20M(self):
        prev = torch_npu.npu.memory_allocated()
        a = torch.rand(1024 * 1024 * 40, dtype=torch.float32).npu()
        # 实际申请1G内存
        version = utils.get_cann_version(module="CANN")
        if (utils.get_soc_version() >= 260 and version >= "9.1.0"):
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((4 * 40 * 1024 * 1024) / 512) * 512)
        else:
            self.assertEqual(torch_npu.npu.memory_allocated(),
                             prev + math.ceil((4 * 40 * 1024 * 1024 + 32) / 512) * 512)

    def test_huge_memory_alloc_512B(self):
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
        prev = torch_npu.npu.memory_allocated()
        a = torch.rand(8 * 8 * 16, dtype=torch.float32).npu()  # 512B
        # 实际申请1M内存
        version = utils.get_cann_version(module="CANN")
        if (utils.get_soc_version() >= 260 and version >= "9.1.0"):
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((8 * 8 * 16 * 4) / 512) * 512)
        else:
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((8 * 8 * 16 * 4 + 32) / 512) * 512)

    def test_huge_memory_alloc_512B_by_vm(self):
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        prev = torch_npu.npu.memory_allocated()
        a = torch.rand(8 * 8 * 16, dtype=torch.float32).npu()  # 512B
        # 实际申请1M内存
        version = utils.get_cann_version(module="CANN")
        if (utils.get_soc_version() >= 260 and version >= "9.1.0"):
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((8 * 8 * 16 * 4) / 512) * 512)
        else:
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((8 * 8 * 16 * 4 + 32) / 512) * 512)
        del os.environ["PYTORCH_NPU_ALLOC_CONF"]

    def test_multi_stream_lazy_reclaim_trigger_event(self):
        code = textwrap.dedent("""\
            import os
            os.environ["PYTORCH_NPU_ALLOC_CONF"] = "multi_stream_lazy_reclaim:True"

            import time
            import torch
            import torch_npu

            max_event_lazy_num = 512
            shared_stream = torch.npu.Stream()
            x_list = []
            for i in range(max_event_lazy_num):
                x_list.append(torch.empty(16, 16, device="npu", dtype=torch.bfloat16))
                x_list[i].record_stream(shared_stream)

            x = torch.empty(16, 16, device="npu", dtype=torch.bfloat16)
            x.record_stream(shared_stream)

            with torch.npu.stream(shared_stream):
                y = x + 0.1

            del x_list
            del x

            time.sleep(0.1)
            dumb = torch.empty(16, 16, device="npu", dtype=torch.bfloat16)
            del dumb
        """)

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"Subprocess failed with return code {result.returncode}.\\n"
            f"stdout: {result.stdout}\\n"
            f"stderr: {result.stderr}"
        )


if __name__ == '__main__':
    run_tests()
