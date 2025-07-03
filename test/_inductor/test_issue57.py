import torch
from torch.testing._internal.common_utils import run_tests
from testutils import TestUtils
import torch_npu


class Test_issue57(TestUtils):
    def op_sum(self, view_12, embedding_1, slice_11):
        permute_7 = torch.ops.aten.permute.default(embedding_1, [2, 0, 1])
        embedding_1 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(permute_7, 0)
        permute_7 = None

        add_5 = torch.ops.aten.add.Tensor(unsqueeze_4, slice_11)
        slice_8 = slice_11 = None
        add_6 = torch.ops.aten.add.Tensor(view_12, add_5)
        view_12 = None
        return add_6

    def test_issue57(self):
        device = 'npu'
        embedding_1 = torch.randn((512, 512, 64), device=device, dtype=torch.float32)
        primals_221 = torch.randn((1, 1, 1, 512), device=device, dtype=torch.float32)
        view_12 = torch.randn((1, 64, 512, 512), device=device, dtype=torch.float32)
        slice_11 = torch.randn((1, 1, 1, 512), device=device, dtype=torch.float32)

        ref = self.op_sum(view_12, embedding_1, primals_221)
        func = torch.compile(self.op_sum, backend="inductor", dynamic=False)
        calc = func(view_12, embedding_1, primals_221)

        self.assertEqual(ref, calc, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
