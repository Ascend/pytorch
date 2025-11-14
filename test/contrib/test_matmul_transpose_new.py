import torch
import torch_npu
import torch_npu.contrib.function._matmul_transpose as matmul_transpose
from torch_npu.testing.testcase import TestCase, run_tests


class TestMatmulTransposeNew(TestCase):
    def test_forward_pass_basic(self):
        tensor1 = torch.randn(2, 3, 4, 5).npu()
        tensor2 = torch.randn(2, 3, 4, 5).npu()
        result = matmul_transpose.MatmulApply.apply(tensor1, tensor2)
        excepted = torch.matmul(tensor1, tensor2.transpose(-2, -1))
        self.assertEqual(result, excepted)


if __name__ == "__main__":
    run_tests()
