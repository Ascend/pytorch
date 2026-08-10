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
    (torch.int8,   "i8"),
    (torch.int32,  "i32"),
    (torch.int64,  "i64"),
]
D2H_TEST_DTYPES = [torch.float16, torch.bfloat16, torch.int8, torch.int32]


def save_tensor(tensor, path, acl_format):
    x = torch_npu.npu_format_cast(tensor.npu(), acl_format)
    torch.save(x, path)


def load_tensor(tensor, path):
    y = torch.load(path)

    if not torch.allclose(y.cpu(), tensor):
        raise ValueError("load tensor not equal to save tensor.")


@unittest.skipIf(IS_ASCEND950,
    "Ascend950 uses aclnn-only path; see TestSerializationFormatAscend950")
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

                self.assertEqual(fmt_before, fmt_after,
                    f"{name}: format {fmt_before} != {fmt_after}")
                self.assertTrue(torch.equal(x.cpu(), y.cpu()),
                    f"{name}: value mismatch")
            finally:
                os.unlink(path)

    def test_nz_d2h_and_repr(self):
        """D2H, repr, str, print on private-format tensor must not crash."""
        for dt in D2H_TEST_DTYPES:
            try:
                x = torch.randn(64, 64, dtype=torch.float32).to(dt).npu()
                x = torch_npu.npu_format_cast(x, torch_npu.Format.FRACTAL_NZ)
            except Exception:
                continue

            c = x.cpu()
            self.assertEqual(c.device.type, "cpu")

            self.assertIsInstance(repr(x), str)
            print(x)


if __name__ == "__main__":
    run_tests()
