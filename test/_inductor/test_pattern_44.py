import torch
from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class Test_pattern_44(TestUtils):
    def forward(self, arg0_1: "f16[1, 64, 33]", arg1_1: "f16[1, 64, 1]"):
        expand: "f16[192, 64, 33]" = torch.ops.aten.expand.default(arg0_1, [192, -1, -1]);
        slice_3: "f16[192, 64, 32]" = torch.ops.aten.slice.Tensor(expand, 2, 1, 9223372036854775807)
        slice_6: "f16[192, 64, 1]" = torch.ops.aten.slice.Tensor(expand, 2, -1, 9223372036854775807);
        mul: "f16[192, 64, 1]" = torch.ops.aten.mul.Tensor(slice_6, arg1_1);
        slice_8: "f16[192, 32, 32]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 32)
        sum_1: "f16[192, 32]" = torch.ops.aten.sum.dim_IntList(slice_8, [1]);
        cat: "f16[192, 320]" = torch.ops.aten.cat.default([sum_1], 1);
        return (mul, cat)

    def test_pattern_44(self):
        arg0 = torch.randn((1, 64, 33), device='npu').to(torch.float16)
        arg1 = torch.randn((1, 64, 1), device='npu').to(torch.float16)

        compiled_net = torch.compile(self.forward, backend="inductor")
        output = self.forward(arg0, arg1)
        output1 = compiled_net(arg0, arg1)
        self.assertEqual(output[0], output1[0], atol=1e-3, rtol=1e-3)
        self.assertEqual(output[1], output1[1], atol=1e-3, rtol=1e-3)

instantiate_parametrized_tests(Test_pattern_44)

if __name__ == "__main__":
    run_tests()