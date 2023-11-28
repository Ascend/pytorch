import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMv(TestCase):
    def cpu_op_exec(self, input1, input2):
        cpu_output = torch.mv(input1, input2)
        cpu_output = cpu_output.numpy()
        return cpu_output

    def npu_op_exec(self, input1, input2):
        npu_output = torch.mv(input1, input2)
        npu_output = npu_output.cpu()
        npu_output = npu_output.numpy()
        return npu_output

    def npu_op_exec_out(self, input1, input2, output):
        torch.mv(input1, input2, out=output)
        output = output.cpu()
        output = output.numpy()
        return output

    def test_mv_shape_format(self):
        shape_format = [
            [[np.float16, -1, (3, 3)], [np.float16, -1, (3)]],
            [[np.float16, -1, (5, 8)], [np.float16, -1, (8)]],
            [[np.float16, -1, (8, 9)], [np.float16, -1, (9)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1.float(), cpu_input2.float())
            npu_output = self.npu_op_exec(npu_input1.float(), npu_input2.float())
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3, prec16=1.e-3)

    def test_mv_out_shape_format(self):
        shape_format = [
            [[np.float16, -1, (3, 3)], [np.float16, -1, (3)], [np.float16, -1, (3)]],
            [[np.float16, -1, (5, 8)], [np.float16, -1, (8)], [np.float16, -1, (5)]],
            [[np.float16, -1, (8, 9)], [np.float16, -1, (9)], [np.float16, -1, (8)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -100, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[2], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1.float(), cpu_input2.float())
            npu_output = self.npu_op_exec_out(npu_input1.float(), npu_input2.float(), npu_input3.float())
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3, prec16=1.e-3)

    def test_mv_with_transpose(self):
        cpu_mat = torch.rand(1, 256).half().float()
        npu_mat = cpu_mat.npu()
        cpu_vec = torch.tensor([1.]).half().float()
        npu_vec = cpu_vec.npu()

        cpu_mv = torch.mv(cpu_mat.t(), cpu_vec)
        npu_mv = torch.mv(npu_mat.t(), npu_vec)
        self.assertRtolEqual(cpu_mv, npu_mv.cpu())


if __name__ == "__main__":
    run_tests()
