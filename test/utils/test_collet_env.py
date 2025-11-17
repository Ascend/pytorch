import os
import site
import tempfile
import torch

from torch.utils import collect_env as torch_collect_env

from torch_npu.testing.testcase import TestCase, run_tests

try:
    import torch_npu

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

try:
    import torch_npu
    TORCH_NPU_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_NPU_AVAILABLE = False

from torch_npu.utils.collect_env import (
    SystemEnv,
    get_torch_npu_install_path,
    check_path_owner_consistent,
    check_directory_path_readable,
    get_torch_npu_version,
    get_env_info,
    pretty_str
)


class TestCollectEnv(TestCase):

    def test_pretty_str_with_none_and_empty(self):
        env_info = SystemEnv(
            torch_version='1.0.0',
            torch_npu_version='1.0.0',
            is_debug_build='False',
            gcc_version='9.3.0',
            clang_version='10.0.0',
            cmake_version='3.16.0',
            os="Linux",
            libc_version='2.27',
            python_version='3.8.0 (64-big runtime)',
            python_platform='Linux',
            pip_version='pip',
            pip_packages='',
            conda_packages='',
            caching_allocator_config='default',
            is_xnnpack_available=True,
            cpu_info='Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz',
            cann_version='not known'
        )

        result = pretty_str(env_info)
        self.assertIn('PyTorch version: 1.0.0', result)
        self.assertIn('CANN:', result)
        self.assertIn('not known', result)

    def test_check_directory_path_readable_symlink(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            symlink_path = os.path.join(temp_dir, "symlink_to_dir")
            os.symlink(temp_dir, symlink_path)

            with self.assertRaises(RuntimeError) as context:
                check_directory_path_readable(symlink_path)
            self.assertIn("Invalid path is a soft chain", str(context.exception))

    def test_get_env_info_torch_not_available(self):
        import torch_npu.utils.collect_env as collect_env_module
        original_torch_available = collect_env_module.TORCH_AVAILABLE
        original_torch = collect_env_module.torch

        collect_env_module.TORCH_AVAILABLE = False
        collect_env_module.torch = None

        try:
            result = get_env_info()
            self.assertEqual(result.torch_version, 'N/A')
            self.assertEqual(result.is_debug_build, 'N/A')
        finally:
            collect_env_module.TORCH_AVAILABLE = original_torch_available
            collect_env_module.torch = original_torch

    def test_get_torch_npu_varsion_npu_availabel(self):
        import torch_npu.utils.collect_env as collect_env_module
        original_torch_npu_available = collect_env_module.TORCH_NPU_AVAILABLE
        original_torch_npu = collect_env_module.torch_npu

        collect_env_module.TORCH_NPU_AVAILABLE = False
        collect_env_module.torch_npu = None

        try:
            result = get_torch_npu_version()
            self.assertEqual(result, 'N/A')
        finally:
            collect_env_module.TORCH_NPU_AVAILABLE = original_torch_npu_available
            collect_env_module.torch_npu = original_torch_npu

    def test_check_path_owner_consistent_nonexistent_path(self):
        with self.assertRaises(RuntimeError) as context:
            check_path_owner_consistent("/non/existent/path")
        self.assertIn("The path does not exist", str(context.exception))

    def test_get_torch_npu_install_path_empty_site_packages(self):
        original_getsitepackages = site.getsitepackages

        def mock_getsitepackages():
            return []

        site.getsitepackages = mock_getsitepackages

        try:
            result = get_torch_npu_install_path()
            self.assertEqual(result, "")
        finally:
            site.getsitepackages = original_getsitepackages

    def test_pretty_str_multiline_cpu_info(self):
        env_info = SystemEnv(
            torch_version='1.0.0',
            torch_npu_version='1.0.0',
            is_debug_build='False',
            gcc_version='9.3.0',
            clang_version='10.0.0',
            cmake_version='3.16.0',
            os="Linux",
            libc_version='2.27',
            python_version='3.8.0 (64-big runtime)',
            python_platform='Linux',
            pip_version='pip',
            pip_packages='numpy==1.19.0',
            conda_packages='',
            caching_allocator_config='default',
            is_xnnpack_available=True,
            cpu_info='Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz\\nCore(s): 6\\nThread(s) per core: 2',
            cann_version='not known'
        )

        result = pretty_str(env_info)
        self.assertIn('PyTorch version: 1.0.0', result)
        self.assertIn('CANN:', result)
        self.assertIn('not known', result)

    def test_get_env_info_torch_available(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is not available")

        result = get_env_info()
        self.assertEqual(result.torch_version, torch.__version__)
        self.assertEqual(result.is_debug_build, str(torch.version.debug))


if __name__ == "__main__":
    run_tests()