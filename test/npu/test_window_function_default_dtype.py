"""Window factories must resolve the default dtype before building the window.

``torch.hann_window(n, device="npu")`` and its siblings take the dtype from
``torch.get_default_dtype()`` when none is given. The NPU overrides used to
hand the still-unset dtype straight to ``at::arange``, which applies its own
integral default, so the window came out int64: ``window_function_checks``
still passed (an unset dtype reads back as the default float) and the first
elementwise call then failed on device -- ``aclnnInplaceCos`` reporting 161002
for DT_INT64 for hann/hamming/blackman, and a ``Long did not match Float``
type error for bartlett.
"""

import torch

import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests


class TestWindowFunctionDefaultDtype(TestCase):

    FACTORIES = [
        torch.hann_window,
        torch.hamming_window,
        torch.blackman_window,
        torch.bartlett_window,
    ]
    LENGTHS = [0, 1, 2, 7, 16]

    def test_default_dtype_matches_cpu(self):
        for factory in self.FACTORIES:
            for length in self.LENGTHS:
                with self.subTest(factory=factory.__name__, length=length):
                    expected = factory(length)
                    got = factory(length, device="npu")
                    torch.npu.synchronize()
                    self.assertEqual(got.dtype, torch.get_default_dtype())
                    self.assertEqual(got.dtype, expected.dtype)
                    self.assertEqual(got.shape, expected.shape)
                    self.assertRtolEqual(got.cpu(), expected)

    def test_periodic_flag_matches_cpu(self):
        for factory in self.FACTORIES:
            for periodic in (True, False):
                with self.subTest(factory=factory.__name__, periodic=periodic):
                    expected = factory(16, periodic)
                    got = factory(16, periodic, device="npu")
                    torch.npu.synchronize()
                    self.assertEqual(got.shape, expected.shape)
                    self.assertRtolEqual(got.cpu(), expected)

    def test_explicit_dtype_still_honoured(self):
        # Guards the path that already worked: an explicit dtype must win.
        for factory in self.FACTORIES:
            for dtype in (torch.float32, torch.float16):
                with self.subTest(factory=factory.__name__, dtype=dtype):
                    got = factory(16, dtype=dtype, device="npu")
                    torch.npu.synchronize()
                    self.assertEqual(got.dtype, dtype)
                    self.assertRtolEqual(got.cpu(), factory(16, dtype=dtype))

    def test_default_dtype_override_is_followed(self):
        # The resolution must read the *current* default dtype, not float32.
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            for factory in self.FACTORIES:
                with self.subTest(factory=factory.__name__):
                    got = factory(8, device="npu")
                    torch.npu.synchronize()
                    self.assertEqual(got.dtype, torch.float64)
        finally:
            torch.set_default_dtype(previous)

    def test_hamming_alpha_beta_overloads(self):
        # hamming_window has three extra overloads; they share the same body.
        for args in ((16, True, 0.54), (16, True, 0.54, 0.46)):
            with self.subTest(args=args):
                expected = torch.hamming_window(*args)
                got = torch.hamming_window(*args, device="npu")
                torch.npu.synchronize()
                self.assertEqual(got.dtype, expected.dtype)
                self.assertRtolEqual(got.cpu(), expected)


if __name__ == "__main__":
    run_tests()
