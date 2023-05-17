import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestReplicationPad2dBackward(TestCase):

    def npu_op_exec(self, input1, pad):
        m = torch.nn.ReplicationPad2d(pad).to("npu")
        input1.requires_grad = True
        output = m(input1)
        output.backward(torch.ones_like(output))
        npu_grad = input1.grad
        output = output.to("cpu")
        output = output.detach().numpy()
        return output, npu_grad.cpu().numpy()

    def cpu_op_exec(self, input1, pad):
        m = torch.nn.ReplicationPad2d(pad)
        input1.requires_grad = True
        output = m(input1)
        output.backward(torch.ones_like(output))
        cpu_grad = input1.grad
        output = output.detach().numpy()
        return output, cpu_grad.numpy()

    def test_replicationPad2d_backward_shape_format(self):
        shape_format = [
            [[np.float16, 0, (1, 1, 27, 27)], [2, 2, 2, 2]],
            [[np.float16, 0, (1, 1, 27, 27)], 3],
            [[np.float16, 0, (4, 27, 27)], 3],
            [[np.float16, 0, (2, 3, 27, 27)], 0],
            [[np.float16, 0, (1, 1, 27, 27)], [0, 0, 0, 0]],
            [[np.float32, 0, (20, 1, 23, 27)], [2, 2, 2, 2]],
            [[np.float32, 0, (13, 1, 27, 27)], 3],
            [[np.float32, 0, (4, 27, 27)], 3],
            [[np.float32, 0, (21, 3, 27, 27)], 0],
            [[np.float32, 0, (1, 1, 27, 27)], [0, 0, 0, 0]]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            if item[0][0] == np.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input1, item[1])
            npu_output, npu_grad = self.npu_op_exec(npu_input1, item[1])
            if item[0][0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
                cpu_grad = cpu_grad.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
