import unittest

import torch
from torch._inductor import config
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import torch_npu  # noqa: F401


@instantiate_parametrized_tests
@unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
class TestIntegerDivision(TestCase):
    @parametrize("rounding_mode", ["floor", "trunc"])
    @parametrize("divisor", [3, -3])
    @parametrize("divisor_device", ["cpu", "npu"])
    def test_int64_division(self, rounding_mode, divisor, divisor_device):
        x = torch.tensor(
            [-(2**40) - 1, -5, -1, 0, 7, 8, 2**40 + 1],
            device="npu", dtype=torch.int64,
        )
        y = torch.tensor(divisor, device=divisor_device, dtype=torch.int64)

        def fn(x, y):
            return torch.div(x, y, rounding_mode=rounding_mode)

        torch._dynamo.reset()
        try:
            compiled = torch.compile(
                fn, fullgraph=True, options={"npu_backend": "triton_experimental"}
            )
            with fresh_cache(), config.patch("size_asserts", True):
                self.assertEqual(compiled(x, y), fn(x, y))
        finally:
            torch._dynamo.reset()

    def test_fused_int64_division(self):
        def fn(x):
            return torch.floor_divide(x + 1, 3) + 2

        x = torch.tensor([-(2**40), 2**40], device="npu", dtype=torch.int64)
        torch._dynamo.reset()
        try:
            compiled = torch.compile(
                fn, fullgraph=True, options={"npu_backend": "triton_experimental"}
            )
            with fresh_cache(), config.patch("size_asserts", True):
                self.assertEqual(compiled(x), fn(x))
        finally:
            torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
