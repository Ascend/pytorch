import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

fake_mode = FakeTensorMode()


class TestNpuTranspose(TestCase):
    def test_npu_transpose(self):
        with fake_mode:
            npu_input = torch.randn((5, 3, 6, 4)).npu()
            perm = [1, 0, 2, 3]
            exp_shape = npu_input.permute(perm).shape
            result = torch_npu.npu_transpose(npu_input, perm)

            self.assertEqual(result.shape, exp_shape)


if __name__ == "__main__":
    run_tests()
