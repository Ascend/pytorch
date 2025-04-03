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
        if not version_env.startswith("CANN") and version_env >= "8.1.RC1":
            is_match = (re.match("([0-9]+).([0-9]+).RC([0-9]+)", version)
                        or re.match("([0-9]+).([0-9]+).([0-9]+)", version)
                        or re.match("([0-9]+).([0-9]+).T([0-9]+)", version)
                        or re.match("([0-9]+).([0-9]+).RC([0-9]+).alpha([0-9]+)", version))
            self.assertTrue(is_match, f"The env version is {version_env}. The format of cann version {version} is invalid.")

    def test_compare_cann_version(self):
        version_env = get_cann_version_from_env()
        if not version_env.startswith("CANN") and version_env >= "8.1.RC1":
            result = _is_gte_cann_version("8.1.RC1", module="CANN")
            self.assertTrue(result, f"The env version is {version_env}, the result from _is_gte_cann_version is False")
        else:
            with self.assertRaisesRegex(RuntimeError,
                    "When the version 7.0.0 is less than \"8.1.RC1\", this function is not supported"):
                _is_gte_cann_version("7.0.0", "CANN")


if __name__ == "__main__":
    run_tests()