import torch
from torch.testing._internal.common_utils import run_tests
from testutils import TestUtils
import torch_npu


class TestLazyRegister(TestUtils):
    def test_compile_but_not_invoked(self):

        def run(x, y):
            return x + y

        run = torch.compile(run)
        self.assertFalse(torch_npu.utils._dynamo.is_inductor_npu_initialized())
    
    def test_disale_register_inductor_npu(self):
        torch_npu.utils._dynamo.disable_register_inductor_npu()

        def run(x, y):
            return x - y

        run = torch.compile(run)
        x = torch.randn(10, 20).npu()
        y = torch.randn(10, 20).npu()

        with self.assertRaisesRegex(Exception, "Device npu not supported"):
            _ = run(x, y)

        self.assertFalse(torch_npu.utils._dynamo.is_inductor_npu_initialized())

        torch_npu.utils._dynamo.enable_register_inductor_npu()


if __name__ == "__main__":
    run_tests()
