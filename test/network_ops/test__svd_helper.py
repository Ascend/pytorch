import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSvdHelper(TestCase):
    def cpu_op_exec(self, input1, some, compute_uv=False):
        output_u, output_s, output_v = torch.svd(input1, some, compute_uv)
        return output_u, output_s, output_v

    def npu_op_exec(self, input1, some, compute_uv=False):
        output_u, output_s, output_v = torch.svd(input1, some, compute_uv)
        output_u = output_u.cpu()
        output_s = output_s.cpu()
        output_v = output_v.cpu()
        return output_u, output_s, output_v

    def test_svd_fp32(self):
        shape_format = [
            [[np.float32, -1, [5, 3]]],
            [[np.float32, -1, [2, 3, 4]]],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)

            cpu_u, cpu_s, cpu_v = self.cpu_op_exec(cpu_input, some=True)
            npu_u, npu_s, npu_v = self.npu_op_exec(npu_input, some=True)
            self.assertRtolEqual(cpu_u, npu_u)
            self.assertRtolEqual(cpu_s, npu_s)
            self.assertRtolEqual(cpu_v, npu_v)

            cpu_u, cpu_s, cpu_v = self.cpu_op_exec(cpu_input, some=False)
            npu_u, npu_s, npu_v = self.npu_op_exec(npu_input, some=False)
            self.assertRtolEqual(cpu_u, npu_u)
            self.assertRtolEqual(cpu_s, npu_s)
            self.assertRtolEqual(cpu_v, npu_v)


if __name__ == "__main__":
    run_tests()
