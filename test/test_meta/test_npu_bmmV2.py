import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

fake_mode = FakeTensorMode()


class TestNpuBmmV2(TestCase):
    def test_npu_bmmV2(self):
        with fake_mode:
            npu_input1 = torch.randn(10, 3, 4).npu()
            npu_input2 = torch.randn(10, 4, 5).npu()
            output_size = []
            result = torch_npu.npu_bmmV2(npu_input1, npu_input2, output_size)

            self.assertEqual(result.dtype, npu_input1.dtype)
            self.assertEqual(result.shape, torch.matmul(npu_input1, npu_input2).shape)


if __name__ == "__main__":
    run_tests()
