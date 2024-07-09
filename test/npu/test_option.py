import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestOption(TestCase):

    def test_option_pm(self):
        option = {"ACL_PRECISION_MODE": "allow_fp32_to_fp16"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_osim(self):
        option = {"ACL_OP_SELECT_IMPL_MODE": "high_precision"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_ofi(self):
        option = {"ACL_OPTYPELIST_FOR_IMPLMODE": "Conv2d"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_odl(self):
        option = {"ACL_OP_DEBUG_LEVEL": "2"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_occm(self):
        option = {"ACL_OP_COMPILER_CACHE_MODE": "enable"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_dd(self):
        option = {"ACL_DEBUG_DIR": "test1"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_occd(self):
        option = {"ACL_OP_COMPILER_CACHE_DIR": "test"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_an(self):
        option = {"ACL_AICORE_NUM": "1"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_pme(self):
        option = {"ACL_PRECISION_MODE": "500"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_osime(self):
        option = {"ACL_OP_SELECT_IMPL_MODE": "100"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_dle(self):
        option = {"ACL_OP_DEBUG_LEVEL": "300"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_occme(self):
        option = {"ACL_OP_COMPILER_CACHE_MODE": "2"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_ane(self):
        option = {"ACL_AICORE_NUM": "at"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_fa(self):
        option = {"FORCE_ACLNN_OP_LIST": "index"}
        self.assertIsNone(torch.npu.set_option(option))

if __name__ == "__main__":
    run_tests()
