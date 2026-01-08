import torch
from torch_npu.npu import device_count
from torch_npu.utils._dynamo_device import NpuInterface, current_device, set_device
from torch_npu.utils._inductor import NPUDeviceOpOverrides
from torch_npu._inductor.config import config as npu_config
from torch_npu._inductor.npu_device import NewNPUDeviceOpOverrides
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuDevice(TestCase):
    def test_aoti_get_stream(self):
        overrides = NewNPUDeviceOpOverrides()
        result = overrides.aoti_get_stream()
        excepted = "aoti_torch_get_current_cuda_stream"
        self.assertEqual(result, excepted)

    def test_cpp_stream_type(self):
        overrides = NewNPUDeviceOpOverrides()
        result = overrides.cpp_stream_type()
        excepted = "aclrtStream"
        self.assertEqual(result, excepted)

    def test_abi_compatible_header(self):
        overrides = NewNPUDeviceOpOverrides()
        result = overrides.abi_compatible_header()
        self.assertIn("#include <fstream>", result)
        self.assertIn("#include <vector>", result)
        self.assertIn("#include <iostream>", result)
        self.assertIn("#include <string>", result)
        self.assertIn("#include <tuple>", result)
        self.assertIn("#include <unordered_map>", result)
        self.assertIn("#include <memory>", result)
        self.assertIn("#include <filesystem>", result)
        self.assertIn("#include <assert.h>", result)
        self.assertIn("#include <stdbool.h>", result)
        self.assertIn("#include <sys/syscall.h>", result)
        self.assertIn("#include <torch_npu/csrc/framework/OpCommand.h>", result)
        self.assertIn("#include <torch_npu/csrc/core/npu/NPUStream.h>", result)
        self.assertIn("#include \"runtime/runtime/rt.h\"", result)

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

    def test_import_get_raw_stream_as(self):
        overrides = NewNPUDeviceOpOverrides()
        result = overrides.import_get_raw_stream_as("test_name")
        excepted = "from torch_npu._inductor import get_current_raw_stream as test_name"
        self.assertEqual(result, excepted)


if __name__ == "__main__":
    run_tests()