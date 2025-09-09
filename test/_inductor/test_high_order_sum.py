import torch
from torch.testing._internal.common_utils import run_tests
from testutils import TestUtils
import torch_npu


class TestSum(TestUtils):
    def op_sum(self, npu_dropout_backward_9):
        view_337: "f32[32768, 256]" = torch.ops.aten.view.default(npu_dropout_backward_9, [32768, 256])
        sum_63: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True)
        view_338: "f32[256]" = torch.ops.aten.view.default(sum_63, [256])
        return view_338


    def test_high_order_sum(self):
        npu_dropout_backward_9 = torch.randn((32768, 256), device='npu', dtype=torch.float32)
        ref = self.op_sum(npu_dropout_backward_9)
        func = torch.compile(self.op_sum, backend="inductor")
        calc = func(npu_dropout_backward_9)

        self.assertEqual(ref, calc, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
