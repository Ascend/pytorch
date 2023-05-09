import expecttest

import torch
import numpy as np

import torch_npu
import torch_npu._C
import torch_npu.npu.utils as utils
from torch_npu.testing.testcase import TestCase, run_tests


class TestCheckOverFlow(TestCase):

    def test_check_over_flow(self):
        soc_version = utils.get_soc_version()
        a = torch.Tensor([np.inf, np.inf]).npu()
        a = a + a
        if (soc_version < 220):
            rtn = utils.npu_check_over_flow(a)
            self.assertTrue(rtn == utils.get_npu_overflow_flag())
        else:
            rtn = utils.npu_check_over_flow(a)
            self.assertTrue(rtn)


if __name__ == "__main__":
    run_tests()
