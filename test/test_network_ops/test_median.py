import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMedian(TestCase):

    def cpu_op_exec(self, input1):
        input1 = input1.float()
        output1 = torch.median(input1)
        output1 = output1.half().numpy()
        return output1

    def npu_op_exec(self, input1):
        output1 = torch.median(input1)
        output1 = output1.to("cpu").numpy()
        return output1

    def cpu_op_exec_dim(self, input1, dim, keepdim):
        input1 = input1.float()
        output1, output2 = torch.median(input1, dim, keepdim)
        output1 = output1.half().numpy()
        output2 = output2.numpy()
        return output1, output2

    def npu_op_exec_dim(self, input1, dim, keepdim):
        output1, output2 = torch.median(input1, dim, keepdim)
        output1 = output1.to("cpu").numpy()
        output2 = output2.to("cpu").numpy()
        return output1, output2

    def npu_op_exec_dim_out(self, input1, dim, keepdim, input2, input3):
        torch.median(input1, dim, keepdim, out=(input2, input3))
        output1 = input2.to("cpu").numpy()
        output2 = input3.to("cpu").numpy()
        return output1, output2

    def test_median_shape_format(self):
        shape_format = [
            [np.float16, -1, (10,)],
            [np.float16, 3, (4, 4, 4)],
            [np.float16, 2, (64, 63)],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_median_dim_shape_format(self):
        shape_format = [
            [[np.float16, -1, (10,)], 0, False],
            [[np.float16, 0, (1, 2, 3, 4)], 1, False],
            [[np.float16, -1, (64, 63)], -1, True],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            npu_input2 = torch.empty(0).npu().to(cpu_input1.dtype)
            npu_input3 = torch.empty(0).npu().long()
            cpu_output1, cpu_output2 = self.cpu_op_exec_dim(cpu_input1, item[1], item[2])
            npu_output1, npu_output2 = self.npu_op_exec_dim(npu_input1, item[1], item[2])
            npu_output1_out, npu_output2_out = self.npu_op_exec_dim_out(npu_input1, item[1], item[2], npu_input2,
                                                                        npu_input3)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)
            self.assertRtolEqual(npu_output1_out, npu_output1)
            self.assertRtolEqual(npu_output2_out, npu_output2)


if __name__ == "__main__":
    run_tests()
