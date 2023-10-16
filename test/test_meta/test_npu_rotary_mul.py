import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

fake_mode = FakeTensorMode()


class TestNpuRotaryMul(TestCase):
    def test_npu_rotary_mul(self):
        with fake_mode:
            embedding = torch.randn(2, 8192, 5, 125, dtype=torch.float16, requires_grad=True).npu()
            cosine = torch.randn(1, 8192, 1, 128, dtype=torch.float16, requires_grad=True).npu()
            sine = torch.randn(1, 8192, 1, 128, dtype=torch.float16, requires_grad=True).npu()
            ret = torch.ops.npu.npu_rotary_mul(embedding, cosine, sine)

            self.assertEqual(embedding.shape, ret.shape)
            self.assertEqual(embedding.dtype, ret.dtype)


if __name__ == "__main__":
    run_tests()
