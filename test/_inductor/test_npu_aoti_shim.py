import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_PACKAGES_DIR = REPO_ROOT / "build" / "packages"

# Avoid backend autoload importing the source-tree torch_npu package before
# torch is fully initialized. Prefer the freshly built package output when it
# exists, and otherwise fall back to the installed package.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
for path in ("", str(REPO_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
if BUILD_PACKAGES_DIR.exists():
    sys.path.insert(0, str(BUILD_PACKAGES_DIR))

import torch

from torch_npu.testing.testcase import run_tests, TestCase  # noqa: F401


CPP_EXTENSIONS_DIR = REPO_ROOT / "test" / "cpp_extensions"
sys.path.insert(0, str(CPP_EXTENSIONS_DIR))

from torch_test_cpp_extension.load_npu_aoti_shim import load_npu_aoti_shim_extension


class TestNpuAOTIShim(TestCase):
    module = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.npu.is_available():
            raise RuntimeError("test_npu_aoti_shim requires an available NPU")
        cls.module = load_npu_aoti_shim_extension()

    def test_npu_raw_stream_matches_python_stream(self):
        device_index = 0
        x = torch.randn(8, device=f"npu:{device_index}", dtype=torch.float32)

        with torch.npu.device(device_index):
            expected_default = torch.npu.current_stream().npu_stream
            actual_default = self.module.get_npu_raw_stream(x)
            self.assertEqual(actual_default, expected_default)

            custom_stream = torch.npu.Stream()
            with torch.npu.stream(custom_stream):
                actual_custom = self.module.get_npu_raw_stream(x)
                self.assertEqual(actual_custom, custom_stream.npu_stream)

    def test_zero_size_tensor_from_blob_uses_null_data_path_on_npu(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        result = self.module.make_zero_size_blob_tensor(x)

        self.assertEqual(result.device, x.device)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.layout, torch.strided)
        self.assertEqual(tuple(result.shape), (0,))
        self.assertEqual(tuple(result.stride()), (1,))
        self.assertEqual(result.numel(), 0)

    def test_zero_size_tensor_from_blob_uses_cpu_device_branch(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        result = self.module.make_zero_size_cpu_blob_tensor(x)

        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.layout, torch.strided)
        self.assertEqual(tuple(result.shape), (0,))
        self.assertEqual(tuple(result.stride()), (1,))
        self.assertEqual(result.numel(), 0)

    def test_mkldnn_blob_tensor_v2_is_rejected_by_original_npu_logic(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_mkldnn_blob_tensor_v2_rejected(x), 1)

    def test_blob_tensor_v2_propagates_inner_failure(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(
            self.module.check_blob_tensor_v2_propagates_invalid_device_failure(x),
            1,
        )

    def test_null_delete_paths_accept_nullptr_inputs(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_null_delete_paths(x), 1)

    def test_invalid_stream_guard_path_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_invalid_stream_guard_path(x), 1)

    def test_null_stream_guard_path_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_null_stream_guard_path(x), 1)

    def test_invalid_device_guard_creation_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_invalid_device_guard_creation(x), 1)

    def test_invalid_device_guard_set_index_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_invalid_device_guard_set_index(x), 1)

    def test_invalid_device_current_stream_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_invalid_device_current_stream(x), 1)

    def test_null_stream_guard_output_handle_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_null_stream_guard_output_handle(x), 1)

    def test_null_current_stream_output_handle_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_null_current_stream_output_handle(x), 1)

    def test_null_guard_output_handle_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_null_guard_output_handle(x), 1)

    def test_null_allocator_output_handle_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_null_allocator_output_handle(x), 1)

    def test_null_blob_tensor_output_handle_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_null_blob_tensor_output_handle(x), 1)

    def test_null_blob_tensor_v2_output_handle_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_null_blob_tensor_v2_output_handle(x), 1)

    def test_current_device_stream_lookup_uses_negative_one_semantics(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_current_device_stream_lookup(x), 1)

    def test_default_stream_guard_roundtrip_restores_original_stream(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.module.check_default_stream_guard_roundtrip(x), 1)

    def test_run_npu_shim_checks_restores_device_and_stream(self):
        device_count = torch.npu.device_count()
        target_device = 1 if device_count > 1 else 0
        original_device = 0 if target_device != 0 else target_device

        torch.npu.set_device(original_device)
        x = torch.randn(16, device=f"npu:{target_device}", dtype=torch.float32)
        expected = x + 1

        with torch.npu.device(target_device):
            custom_stream = torch.npu.Stream()
            with torch.npu.stream(custom_stream):
                before_stream = torch.npu.current_stream().npu_stream
                result = self.module.run_npu_shim_checks(x)
                after_stream = torch.npu.current_stream().npu_stream

        torch.testing.assert_close(result, expected)
        self.assertEqual(before_stream, custom_stream.npu_stream)
        self.assertEqual(after_stream, custom_stream.npu_stream)
        self.assertEqual(torch.npu.current_device(), original_device)


if __name__ == "__main__":
    run_tests()
