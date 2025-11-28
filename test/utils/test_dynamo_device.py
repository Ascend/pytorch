from unittest.mock import patch
from torch._dynamo.device_interface import caching_worker_current_devices, caching_worker_device_properties

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils._dynamo_device import NpuInterface


class DynamoDevice(TestCase):

    def test_is_bf16_supported_with_emulation(self):
        result = NpuInterface.is_bf16_supported(including_emulation=True)
        self.assertTrue(result)

    def test_worker_get_device_properties_none_no_cache(self):
        if "npu" in caching_worker_device_properties:
            del caching_worker_device_properties["npu"]

        with patch('torch_npu.npu.device_count', return_value=1):
            with patch('torch_npu.utils._dynamo_device.get_device_properties_npu') as mock_get_props:
                mock_get_props.return_value = "mock_device_prop"
                result = NpuInterface.Worker.get_device_properties(None)
                self.assertEqual(result, "mock_device_prop")

    def test_worker_get_device_properties_string(self):
        with self.assertRaises(AssertionError):
            NpuInterface.Worker.get_device_properties("cuda:0")

    def test_worker_current_device_npu_cached(self):
        if "npu" in caching_worker_current_devices:
            del caching_worker_current_devices["npu"]

        with patch('torch_npu.utils._dynamo_device.current_device', return_value=2):
            result = NpuInterface.Worker.current_device()
            self.assertEqual(result, 2)

    def test_worker_current_device_cached(self):
        caching_worker_current_devices["npu"] = 1
        result = NpuInterface.Worker.current_device()
        self.assertEqual(result, 1)

    def test_worker_set_device(self):
        if "npu" in caching_worker_current_devices:
            del caching_worker_current_devices["npu"]

        NpuInterface.Worker.set_device(0)
        self.assertEqual(caching_worker_current_devices["npu"], 0)


if "__main__" == __name__:
    run_tests()