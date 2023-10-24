import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

fake_mode = FakeTensorMode()


class TestNpuDtypeCast(TestCase):
    def test_npu_dtype_cast(self):
        with fake_mode:
            npu_input = torch.randn((2, 3), dtype=torch.float32).npu()
            dst_dtype = torch.float16
            result = torch_npu.npu_dtype_cast(npu_input, dst_dtype)

            self.assertEqual(result.dtype, dst_dtype)
            self.assertEqual(result.shape, npu_input.shape)


if __name__ == "__main__":
    run_tests()
