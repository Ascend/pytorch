import expecttest

import torch
import torch_npu
import torch_npu._C

import torch_npu.npu.utils as utils

from torch_npu.testing.testcase import TestCase, run_tests


class TestSocVersion(TestCase):

    def test_get_soc_version(self):
        soc_version = utils.get_soc_version()
        self.assertTrue(soc_version >= -1)


if __name__ == "__main__":
    run_tests()
