import torch
import numpy as np
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.decorator import graph_mode


class TestMaxPool2dWithIndices(TestCase):
    def cpu_op_exec(self, inputCpu, kernel_size, stride, padding, dilation, ceil_mode):
        dataCpu, argMaxCpu = F.max_pool2d_with_indices(inputCpu, kernel_size = kernel_size,
                             stride = stride, padding = padding, dilation = dilation,
                             ceil_mode = ceil_mode, return_indices=True)
        return dataCpu,argMaxCpu

    def npu_op_exec(self, inputNpu, kernel_size, stride, padding, dilation, ceil_mode):
        dataNpu, argMaxNpu = F.max_pool2d_with_indices(inputNpu, kernel_size = kernel_size,
                             stride = stride, padding = padding, dilation = dilation,
                             ceil_mode = ceil_mode, return_indices=True)
        output1 = dataNpu.to("cpu").detach()
        output2 = argMaxNpu.to("cpu").detach()
        return output1, output2

    @graph_mode
    def test_max_pool2d_with_indices_fp16(self):
        shape_format = [
            [[np.float16, 0, [256, 64, 112, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float16, 0, [1024, 24, 112, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float16, 0, [1024, 24, 56, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float16, 0, [1024, 24, 112, 56]], [3, 3], [2, 2], 1, 1, False],
            [[np.float16, 3, [256, 64, 112, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float16, 3, [1024, 24, 112, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float16, 3, [1024, 24, 56, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float16, 3, [1024, 24, 112, 56]], [3, 3], [2, 2], 1, 1, False],
            [[np.float16, 4, [256, 64, 112, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float16, 4, [1024, 24, 112, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float16, 4, [1024, 24, 56, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float16, 4, [1024, 24, 112, 56]], [3, 3], [2, 2], 1, 1, False]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output, cpu_arg = self.cpu_op_exec(cpu_input, item[1], item[2], item[3], item[4], item[5])
            npu_output, npu_arg = self.npu_op_exec(npu_input, item[1], item[2], item[3], item[4], item[5])
            cpu_output = cpu_output.to(npu_output.dtype)
            cpu_arg = cpu_arg.to(npu_arg.dtype)

            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())

    @graph_mode
    def test_max_pool2d_with_indices_fp32(self):
        shape_format = [
            [[np.float32, 0, [256, 64, 112, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float32, 0, [1024, 24, 112, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float32, 0, [1024, 24, 56, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float32, 0, [1024, 24, 112, 56]], [3, 3], [2, 2], 1, 1, False],
            [[np.float32, 3, [256, 64, 112, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float32, 3, [1024, 24, 112, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float32, 3, [1024, 24, 56, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float32, 3, [1024, 24, 112, 56]], [3, 3], [2, 2], 1, 1, False],
            [[np.float32, 4, [256, 64, 112, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float32, 4, [1024, 24, 112, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float32, 4, [1024, 24, 56, 112]], [3, 3], [2, 2], 1, 1, False],
            [[np.float32, 4, [1024, 24, 112, 56]], [3, 3], [2, 2], 1, 1, False]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output, cpu_arg = self.cpu_op_exec(cpu_input, item[1], item[2], item[3], item[4], item[5])
            npu_output, npu_arg = self.npu_op_exec(npu_input, item[1], item[2], item[3], item[4], item[5])

            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy(), prec=1.e-3)


if __name__ == "__main__":
    run_tests()
