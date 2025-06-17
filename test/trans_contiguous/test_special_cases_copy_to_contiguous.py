import os
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization


class TestSpecialCasesCopyToContiguous(TestCase):
    def test_expand_copy_to_slice_discontiguous_tensor(self, device="npu"):
        dtype_list = [np.bool_, np.int8, np.int16, np.float16, np.float32, np.int32, np.int64]
        index_list = [3, 8, 16, 32]
        shape_format = [
            [i, j] for i in dtype_list for j in index_list
        ]
        for item in shape_format:
            np_input = np.zeros(40).astype(item[0])
            cpu_input = torch.from_numpy(np_input)
            cpu_out = cpu_input
            cpu_out[:item[1]] = 1
            npu_out = cpu_input.npu()
            npu_out[:item[1]] = 1
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())

    def test_select_broadcast_at_same_axis_copy_contiguous(self, device="npu"):
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3, 29]
        shape_list = [[1, 81, 96, 96]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_out = torch.as_strided(cpu_input, (1, 32, 96, 96), (746496, 0, 96, 1), 737280).clone()
            npu_out = torch.as_strided(npu_input, (1, 32, 96, 96), (746496, 0, 96, 1), 737280).clone()
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())

    def test_h2d_copy_discontiguous(self):
        a = torch.randn(256, 320)
        b = a.transpose(-1, -2) # make b NOT contiguous
        self.assertFalse(b.is_contiguous())
        b = b.npu()
        self.assertFalse(b.is_contiguous()) # after to npu, b is still NOT contiguous
        self.assertEqual(b.stride(), (1, 320))


if __name__ == "__main__":
    run_tests()
