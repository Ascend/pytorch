import torch
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu


class TestResizeAs(TestUtils):
    def op_calc(self, x, target, adder):
        return torch.ops.aten.resize_as_.default(x, target) + adder

    def test_default(self):
        shape = (2, 8)
        target_shape = (8, 2)
        x = torch.randn(shape, dtype=torch.float16, device="npu")
        target = torch.randn(target_shape, dtype=torch.float16, device="npu")
        adder = torch.randn(target_shape, dtype=torch.float16, device="npu")
        std_out = self.op_calc(x, target, adder)
        compile_out = torch.compile(self.op_calc, backend='inductor')(x, target, adder)
        self.assertEqual(std_out, compile_out, atol=1e-5, rtol=1e-5)


instantiate_parametrized_tests(TestResizeAs)


if __name__ == "__main__":
    run_tests()
