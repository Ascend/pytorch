import unittest
import torch
import torch.nn as nn
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAvgPool2d(TestCase):
    def cpu_op_exec(self, input1, ceil_mode):
        m = nn.AvgPool2d(3, stride=(6, 5), padding=0, ceil_mode=ceil_mode)
        output = m(input1)
        output = output.detach().numpy()
        return output

    def npu_op_exec(self, input1, ceil_mode):
        m = nn.AvgPool2d(3, stride=(6, 5), padding=0, ceil_mode=ceil_mode).npu()
        output = m(input1)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def test_avg_pool2d_backward_shape_format_fp16(self):
        shape_format = [
            [[np.float16, 0, (1, 3, 147, 147)], True],
            [[np.float16, 0, (1, 3, 147, 147)], True]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 1)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input.float(), item[1]).astype(np.float16)
            npu_output = self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output, prec16=0.002)

    @unittest.skip("skip test_avg_pool2d_backward_shape_format_fp32 now")
    def test_avg_pool2d_backward_shape_format_fp32(self):
        shape_format = [
            [[np.float32, 0, (1, 3, 147, 147)], True],
            [[np.float32, 0, (1, 3, 147, 147)], True]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 1)
            cpu_output = self.cpu_op_exec(cpu_input, item[1])
            npu_output = self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output, 0.0009)

    def test_avg_pool2d_3d_fp32(self):
        cinput = torch.randn(128, 32, 7)
        ninput = cinput.npu()
        cmodel = torch.nn.AvgPool2d((4, 5))
        nmodel = cmodel.npu()
        cpu_output = cmodel(cinput)
        npu_output = nmodel(ninput)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy(), 0.0009)

    @unittest.skip("skip test_avg_pool2d_4d_fp32 now")
    def test_avg_pool2d_4d_fp32(self):
        cinput = torch.randn(18, 43, 12, 400)
        ninput = cinput.npu()
        kernel = 13
        padding = 6
        stride = 10
        ceil_mode = True
        cmodel = torch.nn.AvgPool2d(kernel, stride=stride, padding=padding, ceil_mode=ceil_mode)
        nmodel = cmodel.npu()
        cpu_output = cmodel(cinput)
        npu_output = nmodel(ninput)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy(), 0.0009)


if __name__ == "__main__":
    run_tests()
