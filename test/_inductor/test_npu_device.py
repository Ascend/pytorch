import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.codegen.npu.device_op_overrides import NewNPUDeviceOpOverrides
from torch_npu._compat.inductor import device_to_aten
from torch_npu._compat.version import CURRENT_VERSION
from torch_npu.testing.testcase import TestCase, run_tests
from torch._inductor.codegen.common import get_device_op_overrides


class TestNpuDevice(TestCase):
    def test_uses_gpu_cpp_wrapper(self):
        overrides = NewNPUDeviceOpOverrides()
        self.assertTrue(overrides.uses_gpu_cpp_wrapper())

    def test_aten_device_type(self):
        overrides = NewNPUDeviceOpOverrides()
        self.assertEqual(overrides.aten_device_type(), "at::kPrivateUse1")

    def test_device_to_aten(self):
        overrides = get_device_op_overrides("npu")
        self.assertEqual(overrides.aten_device_type(), "at::kPrivateUse1")
        self.assertEqual(device_to_aten("npu"), "at::kPrivateUse1")
        if CURRENT_VERSION >= (2, 15):
            from torch._inductor.codegen.cpp_utils import (
                device_to_aten as upstream_device_to_aten,
            )

            self.assertEqual(upstream_device_to_aten("npu"), "at::kPrivateUse1")

    def test_aoti_get_stream(self):
        overrides = NewNPUDeviceOpOverrides()
        result = overrides.aoti_get_stream()
        excepted = "aoti_torch_get_current_npu_stream"
        self.assertEqual(result, excepted)

    def test_cpp_stream_type(self):
        overrides = NewNPUDeviceOpOverrides()
        result = overrides.cpp_stream_type()
        excepted = "aclrtStream"
        self.assertEqual(result, excepted)

    def test_cpp_aoti_stream_guard(self):
        overrides = NewNPUDeviceOpOverrides()
        result = overrides.cpp_aoti_stream_guard()
        excepted = "AOTINpuStreamGuard"
        self.assertEqual(result, excepted)

    def test_cpp_aoti_device_guard(self):
        overrides = NewNPUDeviceOpOverrides()
        result = overrides.cpp_aoti_device_guard()
        excepted = "AOTINpuGuard"
        self.assertEqual(result, excepted)

    def test_device_guard(self):
        overrides = NewNPUDeviceOpOverrides()
        result = overrides.device_guard(0)
        excepted = "torch.npu.utils.device(0)"
        self.assertEqual(result, excepted)

    def test_synchronize(self):
        overrides = NewNPUDeviceOpOverrides()
        result = overrides.synchronize()
        excepted = """
                stream = torch.npu.current_stream()
                stream.synchronize()
                """
        self.assertEqual(result, excepted)

    def test_set_device(self):
        overrides = NewNPUDeviceOpOverrides()
        result = overrides.set_device(0)
        excepted = "torch.npu.set_device(0)"
        self.assertEqual(result, excepted)

        overrides = NewNPUDeviceOpOverrides()
        test_name = "test_name_npu"
        result = overrides.import_get_raw_stream_as(test_name)
        expected = f"from torch._C import _npu_getCurrentRawStream as {test_name}"
        import torch_npu
        if hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait"):
            expected = f"from torch_npu._C import _npu_getCurrentRawStreamNoWait as {test_name}"
        self.assertEqual(result, expected)


if __name__ == "__main__":
    run_tests()
