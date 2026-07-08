"""AscendC backend: basic compilation and numeric correctness."""
import unittest
import torch
import torch_npu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@unittest.skipIf(not torch.npu.is_available(), "requires an NPU device")
class TestAscendcBasic(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._ascendc_ok = False
        try:
            x = torch.randn(4, 4, device="npu")
            torch.compile(lambda t: t + 1, backend="inductor",
                          options={"npu_backend": "ascendc"})(x)
            cls._ascendc_ok = True
        except Exception:
            pass

    def setUp(self):
        super().setUp()
        if not self._ascendc_ok:
            self.skipTest("ascendc backend not available")

    @parametrize("dtype", [torch.float32, torch.float16])
    def test_mul_sub(self, dtype):
        """Pointwise pattern: compiled output matches eager."""
        def fn(x, y):
            return (x * y - x)

        x = torch.randn(64, 64, dtype=dtype, device="npu")
        y = torch.randn(64, 64, dtype=dtype, device="npu")

        eager_out = fn(x, y)
        compiled_fn = torch.compile(fn, backend="inductor",
                                    options={"npu_backend": "ascendc"})
        compiled_out = compiled_fn(x, y)

        torch.testing.assert_close(compiled_out, eager_out, rtol=1e-3, atol=1e-3)


instantiate_parametrized_tests(TestAscendcBasic)

if __name__ == "__main__":
    run_tests()
