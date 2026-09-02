# Owner(s): ["module: dynamo"]
import torch
from torch._dynamo.test_case import TestCase


class TestNpuTensorType(TestCase):
    def test_npu_tensor_type_no_args(self):
        """NPU tensor.type() with no arguments should work under torch.compile."""

        @torch.compile(fullgraph=True)
        def fn(x):
            return x.type()

        x = torch.randn(2, 3, device="npu")
        expected = x.type()
        compiled_result = fn(x)
        self.assertEqual(compiled_result, expected)

    def test_cpu_tensor_type_no_args(self):
        """CPU tensor.type() with no arguments should work under torch.compile."""

        @torch.compile(fullgraph=True)
        def fn(x):
            return x.type()

        x = torch.randn(2, 3)
        expected = x.type()
        compiled_result = fn(x)
        self.assertEqual(compiled_result, expected)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
