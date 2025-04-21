import os
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, check_operators_in_prof

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization


class TestAsStridedCopyToContiguous(TestCase):
    def cpu_op_exec(self, input1, size, stride, storage_offset):
        output = torch.as_strided(input1, size, stride, storage_offset).contiguous()
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, size, stride, storage_offset):
        with torch.autograd.profiler.profile(use_device='npu') as prof:
            output = torch.as_strided(input1, size, stride, storage_offset).contiguous()
        self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, [
                         'contiguous_h_combined']) or
                         check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                         True, "Error operators called!")
        output = output.cpu().numpy()
        return output

    def test_as_strided(self):
        dtype_list = [bool, np.int32, np.float16, np.float32, np.int8, np.uint8, np.int64]
        format_list = [-1]
        small_shape_list = [[5, 5]]
        small_shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in small_shape_list
        ]

        for item in small_shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input, (3, 3), (1, 2), 1)
            npu_output = self.npu_op_exec(npu_input, (3, 3), (1, 2), 1)
            self.assertRtolEqual(cpu_output, npu_output)

        other_shape_format = [
            [[np.float16, 0, [13, 23]], (10, 15), (1, 2), 1],
            [[np.float16, 3, [2, 13, 23]], (10, 15), (1, 2), 2],
            [[np.float32, 29, [6, 32, 8, 2]], (8, 6, 2), (5, 4, 1), 3],
        ]

        for item in other_shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input, item[1], item[2], item[3])
            npu_output = self.npu_op_exec(npu_input, item[1], item[2], item[3])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
