import unittest
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNpuSortV2(TestCase):
    def custom_op_exec(self, input1, dim, descending):
        output, _ = torch.sort(input1, dim=dim, descending=descending)
        output = output.cpu().numpy()
        return output

    def npu_op_exec(self, input1, dim, descending):
        output = torch_npu.npu_sort_v2(input1, dim=dim, descending=descending)
        output = output.cpu().numpy()
        return output

    @unittest.skip("skip test_npu_sort_v2 now")
    def test_npu_sort_v2(self):
        shape_format = [
            [[np.float16, 0, (1, 5000)], 0, True],
            [[np.float16, 0, (1, 2, 50000)], 1, False],
            [[np.float16, 0, (1, 289600)], 0, False],
            [[np.float16, 0, (1, 409600)], -1, True],
            [[np.float16, 0, (1, 6, 5)], 1, False],
        ]

        for item in shape_format:
            _, npu_input1 = create_common_tensor(item[0], -100, 100)
            custom_output = self.custom_op_exec(npu_input1, item[1], item[2])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2])

            self.assertRtolEqual(custom_output, npu_output)


if __name__ == "__main__":
    run_tests()
