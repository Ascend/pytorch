import os
import tempfile
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu


# Due to compilation caching, we need to start a new process to load tensor,
# otherwise the cache will be reused without any errors.
# Because torch_npu.testing.testcase.TestCase will set device first and we set device
# in main process then subprocess will raise error, we need a new file without set device
# to test this case.


IS_ASCEND950 = torch_npu._C._npu_get_soc_version() >= 260  # Ascend950 = 260

FORMAT_INFO = {
    "NCHW": 0,
    "NHWC": 1,
    "ND": 2,
    "NC1HWC0": 3,
    "FRACTAL_Z": 4,
    "FRACTAL_NZ": 29,
}

NZ_ROUNDTRIP_DTYPES = [
    (torch.float16, "fp16"),
    (torch.bfloat16, "bf16"),
    (torch.int8, "i8"),
    (torch.int32, "i32"),
    (torch.int64, "i64"),
]
D2H_TEST_DTYPES = [torch.float16, torch.bfloat16, torch.int8, torch.int32]

# Ascend950 materializes a FRACTAL_NZ cast as an NZ_C0 variant (50-54) since C0
# variants are output-only; accept the NZ family when checking the cast result.
NZ_FORMAT_FAMILY = (
    FORMAT_INFO["FRACTAL_NZ"],
    torch_npu.Format.FRACTAL_NZ_C0_16,
    torch_npu.Format.FRACTAL_NZ_C0_32,
    torch_npu.Format.FRACTAL_NZ_C0_2,
    torch_npu.Format.FRACTAL_NZ_C0_4,
    torch_npu.Format.FRACTAL_NZ_C0_8,
)


def save_tensor(tensor, path, acl_format):
    x = torch_npu.npu_format_cast(tensor.npu(), acl_format)
    torch.save(x, path)


def load_tensor(tensor, path):
    y = torch.load(path)

    if not torch.allclose(y.cpu(), tensor):
        raise ValueError("load tensor not equal to save tensor.")


@unittest.skipIf(IS_ASCEND950, "Ascend950 uses aclnn-only path; see TestSerializationFormatAscend950")
class TestSerializationFormat(TestCase):
    def test_save_load_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            tensor = torch.rand(64, 3, 7, 7)

            proc = torch.multiprocessing.get_context("spawn").Process

            for _, acl_format in FORMAT_INFO.items():
                process_save = proc(
                    target=save_tensor,
                    name="save",
                    args=(tensor, path, acl_format),
                )
                process_save.start()
                process_save.join()
                self.assertEqual(process_save.exitcode, 0)

                process_load = proc(
                    target=load_tensor,
                    name="load",
                    args=(tensor, path),
                )
                process_load.start()
                process_load.join()
                self.assertEqual(process_load.exitcode, 0)


@unittest.skipUnless(IS_ASCEND950, "Ascend950 only")
class TestSerializationFormatAscend950(TestCase):

    def test_save_load_nd_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pt')
            tensor = torch.rand(64, 3, 7, 7)

            proc = torch.multiprocessing.get_context("spawn").Process

            process_save = proc(target=save_tensor, name="save",
                                args=(tensor, path, 2))
            process_save.start()
            process_save.join()
            self.assertEqual(process_save.exitcode, 0)

            process_load = proc(target=load_tensor, name="load",
                                args=(tensor, path))
            process_load.start()
            process_load.join()
            self.assertEqual(process_load.exitcode, 0)

    def test_save_load_nz_round_trip_by_dtype(self):
        """Per-dtype FRACTAL_NZ save/load round-trip."""
        for dt, name in NZ_ROUNDTRIP_DTYPES:
            try:
                x = torch.randn(64, 64, dtype=torch.float32).to(dt).npu()
                x = torch_npu.npu_format_cast(x, torch_npu.Format.FRACTAL_NZ)
            except Exception:
                if dt == torch.int64:
                    continue  # i64: not in CANN WEIGHT_DTYPE_SUPPORT_LIST
                raise

            fmt_before = torch_npu.get_npu_format(x)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
                path = f.name
            try:
                torch.save(x, path)
                y = torch.load(path)
                fmt_after = torch_npu.get_npu_format(y)

                self.assertEqual(fmt_before, fmt_after, f"{name}: format {fmt_before} != {fmt_after}")
                self.assertTrue(torch.equal(x.cpu(), y.cpu()), f"{name}: value mismatch")
            finally:
                os.unlink(path)

    def test_nz_d2h_and_repr(self):
        """D2H, repr, str, print on private-format tensor must not crash."""
        for dt in D2H_TEST_DTYPES:
            x = torch.randn(64, 64, dtype=torch.float32).to(dt).npu()
            x = torch_npu.npu_format_cast(x, torch_npu.Format.FRACTAL_NZ)

            c = x.cpu()
            self.assertEqual(c.device.type, "cpu")

            self.assertIsInstance(repr(x), str)
            print(x)


def cast_copy_tensor(tensor, acl_format):
    # Without allow_internal_format the cast silently downgrades internal formats to ND.
    if acl_format != FORMAT_INFO["ND"]:
        torch_npu.npu.config.allow_internal_format = True
    return torch_npu.npu_format_cast(tensor, acl_format)


@unittest.skipIf(IS_ASCEND950, "Ascend950 copy behavior differs; see TestCopyFormatAscend950")
class TestCopyFormat(TestCase):
    """A2/A3: copy_ across all FORMAT_INFO formats in H2D/D2H/D2D directions."""

    def test_copy_formats_h2d_d2h_d2d(self):
        for fmt_name, fmt in FORMAT_INFO.items():
            src_cpu = torch.randn(2, 3, 7, 7)

            # h2d: NPU dst (fmt) <- CPU src
            dst_h2d = cast_copy_tensor(torch.zeros(2, 3, 7, 7).npu(), fmt)
            dst_h2d.copy_(src_cpu)
            self.assertTrue(torch.equal(dst_h2d.cpu(), src_cpu), f"h2d dst={fmt_name}")

            # d2h: CPU dst <- NPU src (fmt)
            src_d2h = cast_copy_tensor(src_cpu.npu(), fmt)
            dst_d2h = torch.zeros(2, 3, 7, 7)
            dst_d2h.copy_(src_d2h)
            self.assertTrue(torch.equal(dst_d2h, src_cpu), f"d2h src={fmt_name}")

            # d2d: dst (fmt) <- src (other format)
            for other_name, other in FORMAT_INFO.items():
                dst_d2d = cast_copy_tensor(torch.zeros(2, 3, 7, 7).npu(), fmt)
                fmt_before = torch_npu.get_npu_format(dst_d2d)
                src_d2d = cast_copy_tensor(src_cpu.npu(), other)
                dst_d2d.copy_(src_d2d)
                self.assertTrue(torch.equal(dst_d2d.cpu(), src_cpu), f"d2d dst={fmt_name} src={other_name}")
                self.assertEqual(torch_npu.get_npu_format(dst_d2d), fmt_before, f"d2d dst={fmt_name} format changed")


@unittest.skipUnless(IS_ASCEND950, "Ascend950 only")
class TestCopyFormatAscend950(TestCase):
    """Ascend950: internal-format copy_ is only supported device-to-host."""

    COPY_TEST_DTYPE = torch.float16

    def test_copy_base_format_h2d_d2h_d2d(self):
        src_cpu = torch.randn(8, 8, dtype=self.COPY_TEST_DTYPE)
        dst_h2d = torch.zeros(8, 8, dtype=self.COPY_TEST_DTYPE).npu()
        dst_h2d.copy_(src_cpu)
        self.assertTrue(torch.equal(dst_h2d.cpu(), src_cpu))

        src_d2h = src_cpu.npu()
        dst_d2h = torch.zeros(8, 8, dtype=self.COPY_TEST_DTYPE)
        dst_d2h.copy_(src_d2h)
        self.assertTrue(torch.equal(dst_d2h, src_cpu))

        dst_d2d = torch.zeros(8, 8, dtype=self.COPY_TEST_DTYPE).npu()
        dst_d2d.copy_(src_d2h)
        self.assertTrue(torch.equal(dst_d2d.cpu(), src_cpu))

    def test_copy_nz_d2h(self):
        for dt in D2H_TEST_DTYPES:
            src_cpu = torch.randn(8, 8).to(dt)
            src_nz = cast_copy_tensor(src_cpu.npu(), FORMAT_INFO["FRACTAL_NZ"])
            self.assertIn(
                torch_npu.get_npu_format(src_nz),
                NZ_FORMAT_FAMILY,
                f"expected FRACTAL_NZ family, got {torch_npu.get_npu_format(src_nz)}")

            dst_cpu = torch.zeros(8, 8, dtype=dt)
            dst_cpu.copy_(src_nz)
            self.assertTrue(torch.equal(dst_cpu, src_cpu), f"d2h {dt}")

    def test_copy_nz_h2d_not_supported(self):
        dst_nz = cast_copy_tensor(torch.zeros(8, 8, dtype=self.COPY_TEST_DTYPE).npu(),
                                  FORMAT_INFO["FRACTAL_NZ"])
        with self.assertRaisesRegex(RuntimeError, "not supported on Ascend950"):
            dst_nz.copy_(torch.randn(8, 8, dtype=self.COPY_TEST_DTYPE))

    def test_copy_nz_d2d_not_supported(self):
        src_nz = cast_copy_tensor(torch.randn(8, 8, dtype=self.COPY_TEST_DTYPE).npu(),
                                  FORMAT_INFO["FRACTAL_NZ"])
        dst_nz = cast_copy_tensor(torch.zeros(8, 8, dtype=self.COPY_TEST_DTYPE).npu(),
                                  FORMAT_INFO["FRACTAL_NZ"])
        with self.assertRaisesRegex(RuntimeError, "not supported on Ascend950"):
            dst_nz.copy_(src_nz)

        dst_nd = torch.zeros(8, 8, dtype=self.COPY_TEST_DTYPE).npu()
        with self.assertRaisesRegex(RuntimeError, "not supported on Ascend950"):
            dst_nd.copy_(src_nz)


if __name__ == "__main__":
    run_tests()
