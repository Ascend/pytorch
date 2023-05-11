import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestReplicationPad1d(TestCase):

    def cpu_op_exec(self, input1, pad):
        m = torch.nn.ReplicationPad1d(pad)
        output = m(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, pad):
        m = torch.nn.ReplicationPad1d(pad)
        output = m(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_out_exec(self, input1, pad, output):
        m = torch._C._nn.replication_pad1d(input1, pad, out=output)
        m = m.numpy()
        return m

    def npu_op_out_exec(self, input1, pad, output):
        m = torch._C._nn.replication_pad1d(input1, pad, out=output)
        m = m.to("cpu")
        m = m.numpy()
        return m

    def test_replicationPad1d_shape_format_fp16(self):
        shape_format = [
            [[np.float16, 0, (1, 2, 4)], [3, 1]],
            [[np.float16, 2, (1, 2, 4)], [3, 1]],
            [[np.float16, 2, (3, 5)], 3]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            cpu_output = cpu_output.astype(np.float16)
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_replicationPad1d_shape_format_fp32(self):
        shape_format = [
            [[np.float32, 0, (1, 2, 4)], [3, 1]],
            [[np.float32, 2, (1, 2, 4)], [3, 1]],
            [[np.float32, 2, (3, 5)], 3]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_replicationPad1d_out_shape_format_fp16(self):
        shape_format = [
            [[np.float16, 0, (2, 17, 20)], [4, 3]],
            [[np.float16, 3, (2, 17, 20)], [4, 3]],
            [[np.float16, 0, (3, 5)], 3]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpuout = torch.randn(1, 3, 3)
            npuout = cpuout.to(npu_input1.dtype).npu()
            cpu_output = self.cpu_op_out_exec(cpu_input1, item[1], cpuout)
            cpu_output = cpu_output.astype(np.float16)
            npu_output = self.npu_op_out_exec(npu_input1, item[1], npuout)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_replicationPad1d_out_shape_format_fp32(self):
        shape_format = [
            [[np.float32, 0, (2, 17, 20)], [4, 3]],
            [[np.float32, 3, (2, 17, 20)], [4, 3]],
            [[np.float32, 0, (3, 5)], 3]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpuout = torch.randn(1, 3, 3)
            npuout = cpuout.to(npu_input1.dtype).npu()
            cpu_output = self.cpu_op_out_exec(cpu_input1, item[1], cpuout)
            npu_output = self.npu_op_out_exec(npu_input1, item[1], npuout)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
