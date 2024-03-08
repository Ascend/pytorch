import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestUpsampleBicubic2dBackward(TestCase):
    def backward_create_scale_shape_format32(self):
        dtype_list = [np.float32]
        format_list = [0, 3]
        shape_list = [(2, 2, 2, 2), (4, 4, 4, 4)]
        size_list = [(2, 2)]

        shape_format = [[[i, j, k], h] for i in dtype_list
                        for j in format_list for k in shape_list
                        for h in size_list]

        return shape_format

    def backward_create_scale_shape_format16(self):
        dtype_list = [np.float16]
        format_list = [0, 3]
        shape_list = [(2, 2, 2, 2), (4, 4, 4, 4)]
        size_list = [(2, 2)]

        shape_format = [[[i, j, k], h] for i in dtype_list
                        for j in format_list for k in shape_list
                        for h in size_list]

        return shape_format

    def cpu_op_scale_exec(self, input1, size):
        input1.requires_grad = True
        output = torch.nn.functional.interpolate(input1, scale_factor=size, mode="bicubic")
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.detach().numpy()
        return output_grad

    def npu_op_scale_exec(self, input1, size):
        input1.requires_grad = True
        output = torch.nn.functional.interpolate(input1, scale_factor=size, mode="bicubic")
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.to("cpu").detach().numpy()
        return output_grad

    @unittest.skip("skip test_upsample_bicubic2d_float16_scale_shape_format now")
    def test_upsample_bicubic2d_common_scale_shape_format(self):
        for item in self.backward_create_scale_shape_format32():
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 255)
            cpu_output = self.cpu_op_scale_exec(cpu_input1, item[1])
            npu_output = self.npu_op_scale_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    @unittest.skip("skip test_upsample_bicubic2d_float16_scale_shape_format now")
    def test_upsample_bicubic2d_float16_scale_shape_format(self):
        def cpu_op_exec_fp16(input1, size):
            input1 = input1.to(torch.float32)
            input1.requires_grad = True
            output = torch.nn.functional.interpolate(input1, scale_factor=size, mode="bicubic")
            output.backward(torch.ones_like(output))
            output_grad = input1.grad
            output_grad = output_grad.detach().numpy()
            output_grad = output_grad.astype(np.float16)
            return output_grad

        for item in self.backward_create_scale_shape_format16():
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 255)
            cpu_output = cpu_op_exec_fp16(cpu_input1, item[1])
            npu_output = self.npu_op_scale_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
