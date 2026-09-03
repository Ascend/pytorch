# Owner(s): ["module: npu"]

import os
import unittest
from unittest.mock import patch

import torch_npu
from torch_npu.testing.testcase import run_tests


class TestNPUIsAvailable(unittest.TestCase):
    @patch.dict(os.environ, {"PYTORCH_HAL_BASED_NPU_CHECK": "1"})
    def test_hal_based_path(self):
        """Use device_count only when the HAL availability check is enabled."""
        for device_count, expected in ((0, False), (2, True)):
            with self.subTest(device_count=device_count):
                with patch.object(
                    torch_npu.npu, "device_count", return_value=device_count
                ) as mock_device_count:
                    with patch.object(
                        torch_npu._C, "_npu_getDeviceCount"
                    ) as mock_runtime_count:
                        self.assertEqual(torch_npu.npu.is_available(), expected)
                        mock_device_count.assert_called_once_with()
                        mock_runtime_count.assert_not_called()

    def test_runtime_path(self):
        """Use the Runtime API when the HAL availability check is not enabled."""
        for env_value in (None, "0", "true"):
            for device_count, expected in ((0, False), (2, True)):
                with self.subTest(
                    env_value=env_value, device_count=device_count
                ):
                    with patch.dict(os.environ, {}, clear=False):
                        if env_value is None:
                            os.environ.pop("PYTORCH_HAL_BASED_NPU_CHECK", None)
                        else:
                            os.environ["PYTORCH_HAL_BASED_NPU_CHECK"] = env_value

                        with patch.object(
                            torch_npu._C,
                            "_npu_getDeviceCount",
                            return_value=device_count,
                        ) as mock_runtime_count:
                            with patch.object(
                                torch_npu.npu, "device_count"
                            ) as mock_device_count:
                                self.assertEqual(
                                    torch_npu.npu.is_available(), expected
                                )
                                mock_runtime_count.assert_called_once_with()
                                mock_device_count.assert_not_called()


if __name__ == "__main__":
    run_tests()
