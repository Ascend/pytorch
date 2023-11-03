import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestReflectionPad1d(TestCase):

    def cpu_op_out_exec(self, input1, pad, output):
        m = torch._C._nn.reflection_pad1d(input1, pad, out=output)
        m = m.numpy()
        return m

    def npu_op_out_exec(self, input1, pad, output):
        m_n = torch._C._nn.reflection_pad1d(input1, pad, out=output)
        m_n = m_n.to("cpu")
        m_n = m_n.numpy()
        return m_n

    def cpu_op_exec(self, input1, pad):
        m = torch.nn.ReflectionPad1d(pad)
        output = m(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, pad):
        m = torch.nn.ReflectionPad1d(pad)
        output = m(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_reflection_pad1d_out_shape_format_fp16(self):
        shape_format = [
            [[np.float16, 2, (1, 2, 4)], [3, 1]],
            [[np.float16, 3, (1, 2, 4)], [3, 1]],
            [[np.float16, 2, (3, 5)], 3]
        ]

        def cpu_op_out_exec_fp16(input1, pad, output):
            input1 = input1.to(torch.float32)
            m = torch._C._nn.reflection_pad1d(input1, pad, out=output)
            m = m.numpy()
            m = m.astype(np.float16)
            return m

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpuout = torch.randn(1, 3, 3)
            npuout = cpuout.to(npu_input1.dtype).npu()
            cpu_output = cpu_op_out_exec_fp16(cpu_input1, item[1], cpuout)
            npu_output = self.npu_op_out_exec(npu_input1, item[1], npuout)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_reflection_pad1d_out_shape_format_fp32(self):
        shape_format = [
            [[np.float32, 0, (1, 2, 4)], [3, 1]],
            [[np.float32, 2, (1, 2, 4)], [3, 1]],
            [[np.float32, 2, (3, 5)], 3]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpuout = torch.randn(1, 3, 3)
            npuout = cpuout.to(npu_input1.dtype).npu()
            cpu_output = self.cpu_op_out_exec(cpu_input1, item[1], cpuout)
            npu_output = self.npu_op_out_exec(npu_input1, item[1], npuout)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_reflection_pad1d_shape_format_fp16(self):
        shape_format = [
            [[np.float16, 0, (2, 10, 12)], [4, 3]],
            [[np.float16, 3, (2, 10, 12)], [4, 3]],
            [[np.float16, 0, (3, 5)], 3],
        ]

        def cpu_op_exec_fp16(input1, pad):
            input1 = input1.to(torch.float32)
            m = torch.nn.ReflectionPad1d(pad)
            output = m(input1)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_reflection_pad1d_shape_format_fp32(self):
        shape_format = [
            [[np.float32, 2, (2, 10, 12)], [4, 3]],
            [[np.float32, 2, (2, 10, 12)], [4, 3]],
            [[np.float32, 2, (3, 5)], 3]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
