import itertools
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestTriangularSolveHelper(TestCase):
    def cpu_op_exec(self, input1, input2, upper, transpose, unitriangular):
        output_s, output_a = input1.triangular_solve(input2, upper, transpose, unitriangular)
        return output_s, output_a

    def npu_op_exec(self, input1, input2, upper, transpose, unitriangular):
        output_s, output_a = input1.triangular_solve(input2, upper, transpose, unitriangular)
        output_s = output_s.cpu()
        output_a = output_a.cpu()
        return output_s, output_a

    def test_triangular_solve_helper_fp32(self):
        shape_format = [
            [[np.float32, -1, [2, 3]], [np.float32, -1, [2, 2]]],
            [[np.float32, -1, [3, 2, 3]], [np.float32, -1, [3, 2, 2]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            iter_list = itertools.product([True, False], [True, False], [True, False])
            for upper, transpose, unitriangular in iter_list:
                cpu_s, cpu_a = self.cpu_op_exec(cpu_input1, cpu_input2, upper, transpose, unitriangular)
                npu_s, npu_a = self.npu_op_exec(npu_input1, npu_input2, upper, transpose, unitriangular)
                self.assertRtolEqual(cpu_a, npu_a)
                self.assertRtolEqual(cpu_s, npu_s)

    def test_triangular_solve_out(self):
        a = torch.randn(9, 252, 59, 19).npu()
        b = torch.randn(59, 59).npu()
        c = torch.randn(9, 252, 59, 19).npu()
        d = torch.randn(59, 59).npu()
        out = [c, d]
        output = torch.triangular_solve(a, b.tril().npu(), upper=False, transpose=True, unitriangular=False, out=out)


if __name__ == "__main__":
    run_tests()
