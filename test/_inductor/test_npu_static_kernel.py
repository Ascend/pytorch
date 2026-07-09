import stat
import unittest
from pathlib import Path

import torch_npu
from torch_npu._inductor.npu_static_kernel import StaticKernelCompiler, safe_resolve_output_dir
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuStaticKernel(TestCase):
    def test_safe_resolve_output_dir_dot_dot(self):
        with self.assertRaises(ValueError):
            safe_resolve_output_dir("test/../dir")

    def test_safe_resolve_output_dir_null_byte(self):
        with self.assertRaises(ValueError):
            safe_resolve_output_dir("test/../x00dir")

    def test_safe_resolve_output_dir_permission_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            readonly_dir = Path(tmpdir) / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
            build_path = str(readonly_dir / 'subdir')
            with self.assertRaises(RuntimeError):
                safe_resolve_output_dir(build_path)

    def test_safe_resolve_output_dir_symlink(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir) / "real"
            real_dir.mkdir()
            symlink_dir = Path(tmpdir) / "symlink"
            symlink_dir.symlink_to(real_dir)
            build_path = str(symlink_dir / "subdir")
            with self.assertRaises(ValueError):
                safe_resolve_output_dir(build_path)

    def test_uninstall_static_kernel_no_path(self):
        from torch_npu._inductor.npu_static_kernel import uninstall_static_kernel
        with unittest.mock.patch("torch_npu._inductor.npu_static_kernel._uninstall_path", None):
            uninstall_static_kernel()

    def test_safe_resolve_output_dir_absolute_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            abs_dir = Path(tmpdir) / "test_build"
            abs_dir.mkdir()
            result = safe_resolve_output_dir(str(abs_dir))
            self.assertTrue(result.exists())
            self.assertIn("kernel_aot_optimization_build_outputs", str(result))

    def test_reselect_static_kernel_with_path_not_string(self):
        with self.assertRaisesRegex(RuntimeError, "path must be a string"):
            torch_npu._C._aclnn_reselect_static_kernel_with_path(123)

    def test_reselect_static_kernel_with_path_null_byte(self):
        with self.assertRaisesRegex(RuntimeError, "null byte"):
            torch_npu._C._aclnn_reselect_static_kernel_with_path("test\x00dir")

    def test_reselect_static_kernel_with_path_non_existent(self):
        with self.assertRaisesRegex(RuntimeError, "failed to resolve path"):
            torch_npu._C._aclnn_reselect_static_kernel_with_path("/nonexistent/path")

    def test_reselect_static_kernel_with_path_is_file(self):
        import tempfile
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaisesRegex(RuntimeError, "must be a directory"):
                torch_npu._C._aclnn_reselect_static_kernel_with_path(f.name)

if __name__ == "__main__":
    run_tests()
