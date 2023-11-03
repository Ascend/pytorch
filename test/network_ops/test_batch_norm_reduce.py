import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestBatchNormReduce(TestCase):
    def cuda_op_exec(self, input_data):
        cpu_sum = torch.sum(input_data, dim=[0, 2, 3])
        cpu_square_sum = torch.sum(input_data * input_data, dim=[0, 2, 3])
        return cpu_sum.numpy(), cpu_square_sum.numpy()

    def npu_op_exec(self, *args):
        npu_sum, npu_square_sum = torch_npu.batch_norm_reduce(*args)
        out_sum = npu_sum.cpu().numpy()
        out_square_sum = npu_square_sum.cpu().numpy()
        return out_sum, out_square_sum

    def test_batch_norm_reduce(self):
        np.random.seed(1234)
        shape_format = [
            [[np.float32, -1, [2, 3, 12, 12]], 1e-5],
        ]
        for item in shape_format:
            cpu_input1, npu_inputfp32 = create_common_tensor(item[0], 1, 10)
            cpu_output = self.cuda_op_exec(cpu_input1)
            npu_outputfp32 = self.npu_op_exec(npu_inputfp32, item[-1])

            self.assertRtolEqual(cpu_output[0], npu_outputfp32[0])
            self.assertRtolEqual(cpu_output[1], npu_outputfp32[1], 1e-2)


if __name__ == "__main__":
    run_tests()
