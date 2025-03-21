import time
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.contrib.function import matmul_transpose


class TestMatmulTranspose(TestCase):
    def npu_slow_matmul_transpose_op_exec(self, input1, input2):
        output = input1 @ input2.transpose(-2, -1)
        output.sum().backward()
        return output.cpu().detach().numpy()

    def npu_fast_matmul_transpose_op_exec(self, input1, input2):
        output = matmul_transpose(input1, input2)
        output.sum().backward()
        return output.cpu().detach().numpy()

    def npu_slow_matmul_transpose(self, input1, input2):
        output = self.npu_slow_matmul_transpose_op_exec(input1, input2)

        repeat_time = 100
        torch.npu.synchronize()
        t1 = time.time()
        for _ in range(repeat_time):
            self.npu_slow_matmul_transpose_op_exec(input1, input2)
        torch.npu.synchronize()
        slow_time = (time.time() - t1) / repeat_time * 1000

        return output, slow_time

    def npu_fast_matmul_transpose(self, input1, input2):
        output = self.npu_fast_matmul_transpose_op_exec(input1, input2)

        repeat_time = 100
        torch.npu.synchronize()
        t2 = time.time()
        for _ in range(repeat_time):
            self.npu_fast_matmul_transpose_op_exec(input1, input2)
        torch.npu.synchronize()
        fast_time = (time.time() - t2) / repeat_time * 1000

        return output, fast_time

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_matmul_transpose_shape_format(self):
        shape_format = [
            [[np.float16, 2, [50, 25, 7, 100]], [np.float16, 2, [50, 25, 10, 100]]],
            [[np.float16, 2, [68, 5, 75, 16]], [np.float16, 2, [68, 5, 43, 16]]],
        ]
        for item in shape_format:
            _, mat1_npu = create_common_tensor(item[0], -10, 10)
            _, mat2_npu = create_common_tensor(item[1], -10, 10)
            mat1_npu.requires_grad_(True)
            mat2_npu.requires_grad_(True)
            slow_output, slow_time = \
                self.npu_slow_matmul_transpose(mat1_npu, mat2_npu)
            fast_output, fast_time = \
                self.npu_fast_matmul_transpose(mat1_npu, mat2_npu)

            self.assertRtolEqual(slow_output, fast_output)
            self.assertTrue(slow_time > fast_time)


if __name__ == "__main__":
    run_tests()
