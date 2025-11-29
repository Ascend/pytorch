from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils._inductor import NPUDeviceOpOverrides


class TestInductor():

    def test_device_guard(self):
        overrides = NPUDeviceOpOverrides()
        result = overrides.device_guard(1)
        expected = "torch_npu.npu._DeviceGuard(1)"
        self.assertEqual(result, expected)

    def test_synchronize(self):
        overrides = NPUDeviceOpOverrides()
        result = overrides.synchronize()
        expected = "torch_npu.npu.synchronize()"
        self.assertEqual(result, expected)

    def test_set_device(self):
        overrides = NPUDeviceOpOverrides()
        result = overrides.set_device(0)
        expected = "torch_npu.npu.set_device(0)"
        self.assertEqual(result, expected)

    def test_import_get_raw_stream_as(self):
        overrides = NPUDeviceOpOverrides()
        result = overrides.import_get_raw_stream_as("test_name")
        expected = "from torch._C import _npu_getCurrentRawStream as test_name"
        self.assertEqual(result, expected)


if __name__ == "__main__":
    run_tests()