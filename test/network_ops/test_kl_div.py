import unittest
import torch
import torch.nn.functional as F
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestKlDiv(TestCase):
    def cpu_op_exec(self, input1, input2, reduction):
        output = torch.kl_div(input1, input2, reduction=reduction)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, reduction):
        output = torch.kl_div(input1, input2, reduction=reduction)
        output = output.cpu()
        output = output.numpy()
        return output

    def test_kl_div_shape_format_fp32(self):
        shape_format = [
            [[torch.float32, 0, (192, 8)], [torch.float32, 0, (192, 8)], 1],
            [[torch.float32, 0, (192, 50000)], [torch.float32, 0, (192, 50000)], 1],
            [[torch.float32, 0, (2, 3)], [torch.float32, 0, (2, 3)], 2],
            [[torch.float32, 0, (4, 5)], [torch.float32, 0, (4, 5)], 2],
            [[torch.float32, 0, (2, 3, 3)], [torch.float32, 0, (2, 3, 3)], 2],
        ]
        for item in shape_format:
            x = torch.randn(item[0][2])
            y = torch.randn(item[1][2])
            cpu_input = F.log_softmax(x, dim=-1)
            cpu_target = F.softmax(y, dim=-1)
            npu_input = cpu_input.npu()
            npu_target = cpu_target.npu()
            reduction = item[2]
            cpu_output = self.cpu_op_exec(cpu_input, cpu_target, reduction)
            npu_output = self.npu_op_exec(npu_input, npu_target, reduction)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_kl_div_shape_format_fp16(self):
        shape_format = [
            [[torch.float16, 0, (192, 8)], [torch.float16, 0, (192, 8)], 1],
            [[torch.float16, 0, (192, 50000)], [torch.float16, 0, (192, 50000)], 1],
            [[torch.float16, 0, (2, 3)], [torch.float16, 0, (2, 3)], 2],
            [[torch.float16, 0, (4, 5)], [torch.float16, 0, (4, 5)], 2],
            [[torch.float16, 0, (2, 3, 3)], [torch.float16, 0, (2, 3, 3)], 2],
        ]
        for item in shape_format:
            x = torch.from_numpy(np.random.randn(*item[0][2]))
            y = torch.from_numpy(np.random.randn(*item[1][2]))
            cpu_input1 = F.log_softmax(x, dim=-1).to(item[0][0])
            cpu_target1 = F.softmax(y, dim=-1).to(item[0][0])
            npu_input = cpu_input1.npu()
            npu_target = cpu_target1.npu()
            reduction = item[2]
            cpu_output = self.cpu_op_exec(cpu_input1.to(torch.float32), cpu_target1.to(torch.float32), reduction)
            npu_output = self.npu_op_exec(npu_input, npu_target, reduction)
            self.assertRtolEqual(cpu_output.astype(np.float16), npu_output)

    def test_kl_div_none_shape_format_fp32(self):
        shape_format = [
            [[torch.float32, 0, (192, 8)], [torch.float32, 0, (192, 8)], 0],
            [[torch.float32, 0, (192, 50000)], [torch.float32, 0, (192, 50000)], 0],
            [[torch.float32, 0, (2, 3)], [torch.float32, 0, (2, 3)], 0],
            [[torch.float32, 0, (4, 5)], [torch.float32, 0, (4, 5)], 0],
            [[torch.float32, 0, (2, 3, 3)], [torch.float32, 0, (2, 3, 3)], 0],
        ]
        for item in shape_format:
            x = torch.randn(item[0][2])
            y = torch.randn(item[1][2])
            cpu_input2 = F.log_softmax(x, dim=-1)
            cpu_target2 = F.softmax(y, dim=-1)
            npu_input = cpu_input2.npu()
            npu_target = cpu_target2.npu()
            reduction = item[2]
            cpu_output = self.cpu_op_exec(cpu_input2, cpu_target2, reduction)
            npu_output = self.npu_op_exec(npu_input, npu_target, reduction)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_kl_div_none_shape_format_fp16(self):
        shape_format = [
            [[torch.float16, 0, (192, 8)], [torch.float16, 0, (192, 8)], 0],
            [[torch.float16, 0, (192, 50000)], [torch.float16, 0, (192, 50000)], 0],
            [[torch.float16, 0, (2, 3)], [torch.float16, 0, (2, 3)], 0],
            [[torch.float16, 0, (4, 5)], [torch.float16, 0, (4, 5)], 0],
            [[torch.float16, 0, (2, 3, 3)], [torch.float16, 0, (2, 3, 3)], 0],
        ]
        for item in shape_format:
            x = torch.randn(item[0][2])
            y = torch.randn(item[1][2])
            cpu_input = F.log_softmax(x, dim=-1).to(item[0][0])
            cpu_target = F.softmax(y, dim=-1).to(item[0][0])
            npu_input = cpu_input.npu()
            npu_target = cpu_target.npu()
            reduction = item[2]
            cpu_output = self.cpu_op_exec(cpu_input.to(torch.float32), cpu_target.to(torch.float32), reduction)
            npu_output = self.npu_op_exec(npu_input, npu_target, reduction)
            self.assertRtolEqual(cpu_output.astype(np.float16), npu_output)


if __name__ == "__main__":
    run_tests()
