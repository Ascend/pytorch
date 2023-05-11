import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestFull(TestCase):
    def test_full_shape_format_fp16(self, device="npu"):
        format_list = [0, 3]
        dtype_list = [torch.float32, torch.float16, torch.int32]
        shape_list = [[5, 8], [2, 4, 1, 1], [16]]
        shape_format = [[[np.float16, i, j], k]
                        for i in format_list for j in shape_list for k in dtype_list]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = torch.full(cpu_input.size(), 6, dtype=item[1], device="cpu")
            cpu_output = cpu_output.numpy()
            npu_output = torch.full(npu_input.size(), 6, dtype=item[1], device="npu")
            npu_output = npu_output.to("cpu")
            npu_output = npu_output.numpy()
            self.assertRtolEqual(cpu_output, npu_output)

    def test_full_shape_format_fp32(self, device="npu"):
        format_list = [0, 3]
        dtype_list = [torch.float32, torch.float16, torch.int32]
        shape_list = [[5, 8], [2, 4, 1, 1], [16]]
        shape_format = [[[np.float32, i, j], k]
                        for i in format_list for j in shape_list for k in dtype_list]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = torch.full(cpu_input.size(), 6, dtype=item[1], device="cpu")
            cpu_output = cpu_output.numpy()
            npu_output = torch.full(npu_input.size(), 6, dtype=item[1], device="npu")
            npu_output = npu_output.to("cpu")
            npu_output = npu_output.numpy()
            self.assertRtolEqual(cpu_output, npu_output)

    def test_full_out(self, device="npu"):

        shape_format = [[[np.float32, 0, [5, 8]], torch.float32]]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 0, 100)
            cpu_output = torch.full(cpu_input1.size(), 6, dtype=item[1], device="cpu")
            cpu_output = cpu_output.numpy()
            npu_output = torch.full(npu_input1.size(), 6, dtype=item[1], out=npu_input2, device="npu")
            npu_output = npu_output.to("cpu")
            npu_output = npu_output.numpy()
            self.assertRtolEqual(cpu_output, npu_output)

    def test_full_without_dtype(self):
        fill_values = [5, True, 1.0]
        for value in fill_values:
            cpu_output = torch.full((10,), value, device="cpu")
            cpu_output = cpu_output.numpy()
            npu_output = torch.full((10,), value, device="npu")
            npu_output = npu_output.to("cpu").numpy()
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
