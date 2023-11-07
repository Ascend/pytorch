import torch
import numpy as np
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMaxPool3dWithIndices(TestCase):
    def cpu_op_exec(self, inputCpu, kernel_size, stride, padding, dilation, ceil_mode):
        dataCpu, argMaxCpu = F.max_pool3d_with_indices(inputCpu, kernel_size=kernel_size,
                                                       stride=stride, padding=padding, dilation=dilation,
                                                       ceil_mode=ceil_mode, return_indices=True)
        return dataCpu, argMaxCpu

    def npu_op_exec(self, inputNpu, kernel_size, stride, padding, dilation, ceil_mode):
        dataNpu, argMaxNpu = F.max_pool3d_with_indices(inputNpu, kernel_size=kernel_size,
                                                       stride=stride, padding=padding, dilation=dilation,
                                                       ceil_mode=ceil_mode, return_indices=True)
        output1 = dataNpu.to("cpu").detach()
        output2 = argMaxNpu.to("cpu").detach()
        return output1, output2

    def test_max_pool3d_with_indices(self):
        shape_format = [
            [np.float16, 30, [1, 3, 19, 19, 19], [3, 3, 3], [2, 2, 2], 1, 1, False],
            [np.float16, 30, [3, 3, 124, 112, 112], 3, [2, 2, 2], 1, 1, True],
            [np.float16, 30, [10, 64, 56, 56, 56], 5, 2, [2, 2, 2], 1, True],
            [np.float16, 30, [10, 10, 10, 10, 10], 3, 2, 1, 1, False],
            [np.float16, 30, [64, 10, 124, 56, 64], 3, 2, 0, 1, False],
            [np.float16, 30, [64, 10, 64, 32, 32], 3, 2, 0, 1, True]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[:3], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output, cpu_arg = self.cpu_op_exec(cpu_input, item[3], item[4], item[5], item[6], item[7])
            npu_output, npu_arg = self.npu_op_exec(npu_input, item[3], item[4], item[5], item[6], item[7])
            cpu_output = cpu_output.to(npu_output.dtype)
            cpu_arg = cpu_arg.to(npu_arg.dtype)

            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())


if __name__ == "__main__":
    run_tests()
