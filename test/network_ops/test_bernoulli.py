import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


SEED = 666


class TestBernoulli(TestCase):

    def get_common_tensor(self, item, minValue, maxValue):
        dtype = item[0]
        shape = item[2]
        npu_input = torch.randn(shape).uniform_(minValue, maxValue).npu().to(dtype)
        return npu_input

    def npu_op_exec(self, input1):
        torch.manual_seed(SEED)
        output = torch.bernoulli(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_inplace_tensor_exec(self, input1, p):
        torch.manual_seed(SEED)
        output = input1.bernoulli_(p)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_inplace_float_exec(self, input1, p):
        torch.manual_seed(SEED)
        output = input1.bernoulli_(p)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_p_exec(self, input1, p):
        torch.manual_seed(SEED)
        input1 = input1.bernoulli_(p)
        input1 = input1.to("cpu")
        input1 = input1.numpy()
        return input1

    def test_bernoulli_p(self):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        p_list = [0.2, 0.6]
        shape_format = [
            [torch.float32, i, j, k]
            for i in format_list
            for j in shape_list
            for k in p_list
        ]
        for item in shape_format:
            npu_input = self.get_common_tensor(item, 0, 1)
            npu_expect_output = self.npu_op_p_exec(npu_input, item[3])
            npu_output = self.npu_op_p_exec(npu_input, item[3])
            self.assertEqual(npu_expect_output, npu_output)

    def test_bernoulli_float32(self):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [torch.float32, i, j]
            for i in format_list
            for j in shape_list
        ]
        for item in shape_format:
            npu_input = self.get_common_tensor(item, 0, 1)
            npu_expect_output = self.npu_op_exec(npu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertEqual(npu_expect_output, npu_output)

    def test_bernoulli_float16(self):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [torch.float16, i, j]
            for i in format_list
            for j in shape_list
        ]
        for item in shape_format:
            npu_input = self.get_common_tensor(item, 0, 1)
            npu_expect_output = self.npu_op_exec(npu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertEqual(npu_expect_output, npu_output)

    def test_bernoulli_tensor_p(self):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [torch.float32, i, j]
            for i in format_list
            for j in shape_list
        ]
        for item in shape_format:
            npu_input = self.get_common_tensor(item, 0, 1)
            npu_input_p = self.get_common_tensor(item, 0, 1)
            npu_expect_output = self.npu_op_inplace_tensor_exec(npu_input, npu_input_p)
            npu_output = self.npu_op_inplace_tensor_exec(npu_input, npu_input_p)
            self.assertEqual(npu_expect_output, npu_output)

    def test_bernoulli_float_p(self):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        p = 0.5
        shape_format = [
            [torch.float32, i, j]
            for i in format_list
            for j in shape_list
        ]
        for item in shape_format:
            npu_input = self.get_common_tensor(item, 0, 1)
            npu_expect_output = self.npu_op_inplace_float_exec(npu_input, p)
            npu_output = self.npu_op_inplace_float_exec(npu_input, p)
            self.assertEqual(npu_expect_output, npu_output)


if __name__ == '__main__':
    run_tests()
