import os
import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


# for A2/A3
os.environ["INF_NAN_MODE_FORCE_DISABLE"] = "1"
os.environ["INF_NAN_MODE_ENABLE"] = "1"


class TestFloatStatus(TestCase):

    def test_float_status(self, device="npu"):
        float_tensor = torch.tensor([40000.0], dtype=torch.float16).npu()
        float_tensor = float_tensor + float_tensor

        input1 = torch.zeros(8).npu()
        float_status = torch_npu.npu_alloc_float_status(input1)
        local_float_status = torch_npu.npu_get_float_status(float_status)

        self.assertTrue(local_float_status.cpu()[0] != 0)
        cleared_float_status = torch_npu.npu_clear_float_status(local_float_status)
        input1 = torch.zeros(8).npu()
        float_status = torch_npu.npu_alloc_float_status(input1)
        local_float_status = torch_npu.npu_get_float_status(float_status)
        self.assertTrue(local_float_status.cpu()[0] == 0)


if __name__ == "__main__":
    run_tests()
