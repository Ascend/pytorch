import os
import subprocess
import datetime
from pathlib import Path
import stat
import torch_npu
from torch_npu._inductor.config import log
from torch_npu._inductor.npu_static_kernel import StaticKernelCompiler
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu._inductor.npu_static_kernel import safe_resolve_output_dir


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
        from torch_npu._inductor.npu_static_kernel import _uninstall_path, uninstall_static_kernel
        _uninstall_path = None
        uninstall_static_kernel()

    def test_safe_resolve_output_dir_absolute_path(self):
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            abs_dir = Path(tmpdir) / "test_build"
            abs_dir.mkdir()
            result = safe_resolve_output_dir(str(abs_dir))
            self.assertTrue(result.exists())
            self.assertIn("kernel_aot_optimization_build_outputs", str(result))


if __name__ == "__main__":
    run_tests()
