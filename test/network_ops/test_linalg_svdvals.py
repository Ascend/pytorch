import torch
import numpy as np
from torch import linalg as LA

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLinalgSvdvals(TestCase):

    def cpu_dtype_out_exec(self, data):
        cpu_output = LA.svdvals(data)
        return cpu_output.numpy()

    def npu_dtype_out_exec(self, data):
        npu_output = LA.svdvals(data)
        return npu_output.cpu().numpy()

    def test_linalg_svdvals(self):
        dtype_list = [np.float32]
        for item in dtype_list:
            cpu_input_1, npu_input_1 = create_common_tensor([item, 0, (2, 3, 4)], -100, 100)
            cpu_output_1 = self.cpu_dtype_out_exec(cpu_input_1)
            npu_output_1 = self.npu_dtype_out_exec(npu_input_1)
            self.assertRtolEqual(cpu_output_1, npu_output_1)


if __name__ == "__main__":
    run_tests()
