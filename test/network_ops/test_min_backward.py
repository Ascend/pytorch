import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMinBackward(TestCase):
    def cpu_op_other_exec(self, input1, input2):
        input1.requires_grad = True
        output = torch.min(input1, input2)
        output.backward(torch.ones_like(output))
        out = input1.grad
        return out

    def npu_op_other_exec(self, input1, input2):
        input1.requires_grad = True
        output = torch.min(input1, input2)
        output.backward(torch.ones_like(output))
        out = input1.grad
        out = out.to('cpu')
        return out

    def min_result_other(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 0, 7)
        if cpu_input1.dtype == torch.float16:
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
        cpu_output_other = self.cpu_op_other_exec(cpu_input1, cpu_input2)
        npu_output_other = self.npu_op_other_exec(npu_input1, npu_input2)
        if npu_output_other.dtype == torch.float16:
            npu_output_other = npu_output_other.float()
        self.assertRtolEqual(cpu_output_other, npu_output_other)

    def test_min_other_shape_format_fp16_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp32_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp16_2d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp32_2d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp16_3d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp32_3d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp16_4d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp32_4d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in
                        keepdim_list
                        ]
        self.min_result_other(shape_format)


if __name__ == "__main__":
    run_tests()
