import re

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch_npu
from torch_npu.utils.collect_env import get_cann_version as get_cann_version_from_env
from torch_npu.npu.utils import get_cann_version, _is_gte_cann_version


class TestCANNversion(TestCase):
    def test_get_cann_version(self):
        version_env = get_cann_version_from_env()
        version = get_cann_version(module="CANN")
        if not version_env.startswith("CANN"):
            if version_env >= "8.1.RC1":
                is_match = (re.match("([0-9]+)\.([0-9]+)\.RC([0-9]+)$", version)
                            or re.match("([0-9]+)\.([0-9]+)\.([0-9]+)$", version)
                            or re.match("([0-9]+)\.([0-9]+)\.T([0-9]+)$", version)
                            or re.match("([0-9]+)\.([0-9]+)\.RC([0-9]+)\.alpha([0-9]+)$", version))
                self.assertTrue(is_match, f"The env version is {version_env}. The format of cann version {version} is invalid.")
            else:
                self.assertTrue(version == "", "When verssion_env < '8.1.RC1', the result of get_cann_version is not right.")
        
        version = get_cann_version(module="CAN")
        self.assertTrue(version == "", "When module is invalid, the result of get_cann_version is not right.")
    
    def test_get_driver_version(self):
        try:
            version = get_cann_version(module="DRIVER")
        except UnicodeDecodeError:
            print("Failed to get driver version. Your driver version is too old, or the environment information about the driver may be incomplete.")
            return
        if re.match("([0-9]+)\.([0-9]+)\.RC([0-9]+)\.B([0-9]+)$", version, re.IGNORECASE):
            version = re.sub(".B([0-9]+)", "", version, flags=re.IGNORECASE)
        if re.match("([0-9]+)\.", version):
            if version >= "25.":
                is_match = (re.match("([0-9]+)\.([0-9]+)\.RC([0-9]+)$", version, re.IGNORECASE)
                            or re.match("([0-9]+)\.([0-9]+)\.([0-9]+)$", version)
                            or re.match("([0-9]+)\.([0-9]+)\.RC([0-9]+)\.([0-9]+)$", version, re.IGNORECASE)
                            or re.match("([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)$", version)
                            or re.match("([0-9]+)\.([0-9]+)\.T([0-9]+)$", version, re.IGNORECASE)
                            or re.match("([0-9]+)\.([0-9]+)\.RC([0-9]+)\.beta([0-9]+)$", version, re.IGNORECASE)
                            or re.match("([0-9]+)\.([0-9]+)\.RC([0-9]+)\.alpha([0-9]+)$", version, re.IGNORECASE)
                            )
                self.assertTrue(is_match, f"The format of driver version {version} is invalid.")
            else:
                self.assertTrue(version == "", "When verssion_env < '25.', the result of get_cann_version is not right.")
    def test_compare_cann_version(self):
        version_env = get_cann_version_from_env()
        if not version_env.startswith("CANN") and version_env >= "8.1.RC1":
            result = _is_gte_cann_version("8.1.RC1", module="CANN")
            self.assertTrue(result, f"The env version is {version_env}, the result from _is_gte_cann_version is False")

            tags = get_cann_version(module="CANN")
            major = int(tags[0]) + 1
            result1 = _is_gte_cann_version(f"{major}.0.0", module="CANN")
            result2 = _is_gte_cann_version(f"{major}.0.T10", module="CANN")
            result3 = _is_gte_cann_version(f"{major}.0.RC1.alpha001", module="CANN")
            self.assertTrue(not result1 and not result2 and not result3, "the result from _is_gte_cann_version is not right.")

        else:
            with self.assertRaisesRegex(RuntimeError,
                    "When the version 7.0.0 is less than \"8.1.RC1\", this function is not supported"):
                _is_gte_cann_version("7.0.0", "CANN")


if __name__ == "__main__":
    run_tests()
