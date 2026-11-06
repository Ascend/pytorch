import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import torch_npu
import torch_npu._inductor  # noqa: F401


@unittest.skipUnless(torch.npu.is_available(), "requires NPU")
class TestComplexDtypeConversion(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def assert_npu_equal(self, actual, expected, *, rtol=0, atol=0):
        self.assertEqual(actual.device.type, "npu")
        self.assertEqual(actual.device, expected.device)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        # aclnnIsClose does not support complex64; compare both components on NPU.
        if actual.is_complex():
            pairs = ((actual.real, expected.real), (actual.imag, expected.imag))
        else:
            pairs = ((actual, expected),)
        for result, reference in pairs:
            torch.testing.assert_close(result, reference, rtol=rtol, atol=atol, equal_nan=True)

    @parametrize("src_dtype,dst_dtype", [
        (torch.float32, torch.complex64),
        (torch.float16, torch.complex64),
        (torch.complex64, torch.float32),
        (torch.complex64, torch.complex64),
    ])
    @parametrize("layout", ["contiguous", "transpose", "slice", "empty", "scalar"])
    def test_convert_element_type(self, src_dtype, dst_dtype, layout):
        real = torch.arange(24, device="npu", dtype=torch.float32).reshape(4, 6) - 12
        if src_dtype.is_complex:
            x = torch.view_as_complex(torch.stack((real, real + 1), dim=-1))
        else:
            x = real.to(src_dtype)
        if layout == "transpose":
            x = x.t()
        elif layout == "slice":
            x = x[1::2, 1::2]
        elif layout == "empty":
            x = x[:0]
        elif layout == "scalar":
            x = x[0, 0]
        before = x.clone()

        def convert(value):
            return torch.ops.prims.convert_element_type.default(value, dst_dtype)

        expected = convert(x)
        actual = torch.compile(convert, backend="inductor", fullgraph=True)(x)
        self.assert_npu_equal(actual, expected)
        self.assert_npu_equal(x, before)
        if dst_dtype.is_complex and not src_dtype.is_complex:
            self.assert_npu_equal(actual.imag, torch.zeros_like(actual.real))
        elif src_dtype.is_complex and not dst_dtype.is_complex:
            self.assert_npu_equal(actual, x.real.to(dst_dtype))

    def test_dynamic_real_to_complex(self):
        def convert(x):
            return torch.ops.prims.convert_element_type.default(x, torch.complex64)

        compiled = torch.compile(convert, backend="inductor", fullgraph=True, dynamic=True)
        for rows, cols in ((2, 3), (5, 7)):
            x = torch.arange(rows * cols, device="npu", dtype=torch.float32).reshape(rows, cols)
            actual = compiled(x)
            self.assert_npu_equal(actual, convert(x))
            self.assert_npu_equal(actual.imag, torch.zeros_like(x))

    def test_nonfinite_real_to_complex(self):
        x = torch.tensor(
            [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1.0, -2.0],
            device="npu", dtype=torch.float32,
        )

        def convert(value):
            return torch.ops.prims.convert_element_type.default(value, torch.complex64)

        actual = torch.compile(convert, backend="inductor", fullgraph=True)(x)
        self.assert_npu_equal(actual, convert(x))
        self.assert_npu_equal(actual.imag, torch.zeros_like(x))
        self.assert_npu_equal(torch.signbit(actual.real[:2]), torch.signbit(x[:2]))

    @parametrize("real_output", [False, True])
    @parametrize("transpose", [False, True])
    def test_fft_real_input(self, real_output, transpose):
        def fft(x):
            result = torch.fft.fftn(x)
            return result.real if real_output else result

        x = torch.randn((16, 16, 16), device="npu", dtype=torch.float32)
        if transpose:
            x = x.transpose(0, 2)
        expected = fft(x)
        actual = torch.compile(fft, backend="inductor", fullgraph=True)(x)
        self.assert_npu_equal(actual, expected, rtol=1e-4, atol=1e-4)


instantiate_parametrized_tests(TestComplexDtypeConversion)

if __name__ == "__main__":
    run_tests()
