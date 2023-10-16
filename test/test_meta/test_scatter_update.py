import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

fake_mode = FakeTensorMode()


class TestScatterUpdate(TestCase):
    def test_scatter_update(self):
        with fake_mode:
            src = torch.randn(1, 4, 4096, 256, dtype=torch.float16, requires_grad=True).npu()
            indices = torch.randint(src.shape[2], (src.shape[0],)).to(torch.int64).npu()
            source = torch.randn(1, 4, 1, 256, dtype=torch.float16, requires_grad=True).npu()
            src.requires_grad = True
            source.requires_grad = True
            ret = torch.ops.npu.npu_rotary_mul(src, indices, source)

            self.assertEqual(src.shape, ret.shape)
            self.assertEqual(src.dtype, ret.dtype)


if __name__ == "__main__":
    run_tests()
