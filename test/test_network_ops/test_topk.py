import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestTopK(TestCase):
    def cpu_op_exec(self, input1, k):
        output, indices = torch.topk(input1, k)
        output = output.numpy()
        indices = indices.numpy().astype(np.int32)
        return output, indices

    def npu_op_exec(self, input1, k):
        output, indices = torch.topk(input1, k)
        output = output.to("cpu")
        indices = indices.to("cpu")
        output = output.numpy()
        indices = indices.numpy().astype(np.int32)
        return output, indices

    def npu_op_exec_out(self, input1, k, output, indices):
        torch.topk(input1, k, out=(output, indices))
        output = output.to("cpu").numpy()
        indices = indices.to("cpu").numpy().astype(np.int32)
        return output, indices

    def topk_result(self, shape_format):
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output, cpu_indices = self.cpu_op_exec(cpu_input, 5)
            npu_output, npu_indices = self.npu_op_exec(npu_input, 5)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-1)

    def topk_large_result(self, shape_format):
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 65504)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            k = 4000000
            cpu_output, cpu_indices = self.cpu_op_exec(cpu_input, k)
            npu_output, npu_indices = self.npu_op_exec(npu_input, k)
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_topk_out_result_fp32(self):
        shape_format = [
            [np.float32, 0, [18]],
            [np.float32, 0, [5, 256]],
            [np.float32, 0, [32, 8, 8]],
            [np.float32, 0, [64, 112, 7, 7]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            cpu_input3, npu_input3 = create_common_tensor(item, 0, 100)
            cpu_output, cpu_indices = self.cpu_op_exec(cpu_input1, 5)
            npu_output, npu_indices = self.npu_op_exec_out(npu_input1, 5, npu_input2, npu_input3.to(torch.int64))
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-1)

    def test_topk_out_result_fp16(self):
        shape_format = [
            [np.float16, 0, [18]],
            [np.float16, 0, [5, 256]],
            [np.float16, 0, [32, 8, 8]],
            [np.float16, 0, [64, 112, 7, 7]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            cpu_input3, npu_input3 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output, cpu_indices = self.cpu_op_exec(cpu_input1, 5)
            npu_output, npu_indices = self.npu_op_exec_out(npu_input1, 5, npu_input2, npu_input3.to(torch.int64))
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_topk_shape_format_fp16_1d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [18]] for i in format_list
        ]
        self.topk_result(shape_format)

    def test_topk_shape_format_fp16_large_1d(self):
        format_list = [-1]
        shape_format = [
            [np.float16, i, [104857600]] for i in format_list
        ]
        self.topk_large_result(shape_format)

    def test_topk_shape_format_fp32_1d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [18]] for i in format_list
        ]
        self.topk_result(shape_format)

    def test_topk_shape_format_fp16_2d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [5, 256]] for i in format_list
        ]
        self.topk_result(shape_format)

    def test_topk_shape_format_fp32_2d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [5, 256]] for i in format_list
        ]
        self.topk_result(shape_format)

    def test_topk_shape_format_fp16_3d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [32, 8, 8]] for i in format_list
        ]
        self.topk_result(shape_format)

    def test_topk_shape_format_fp32_3d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [32, 8, 8]] for i in format_list
        ]
        self.topk_result(shape_format)

    def test_topk_shape_format_fp16_4d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [64, 112, 7, 7]] for i in format_list
        ]
        self.topk_result(shape_format)

    def test_topk_shape_format_fp32_4d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [64, 112, 7, 7]] for i in format_list
        ]
        self.topk_result(shape_format)


if __name__ == "__main__":
    run_tests()
