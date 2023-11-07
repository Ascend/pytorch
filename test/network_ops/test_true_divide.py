import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestTrueDivide(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        return npu_input1, npu_input2

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1, input2):
        output = torch.true_divide(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.true_divide(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar(self, input1, input2):
        input1 = input1.to("npu")
        output = torch.true_divide(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    # pt-int
    def test_true_divide_int32_broadcast(self):
        npu_input1 = self.generate_single_data(0, 100, (2, 2), np.int32)
        npu_input2 = self.generate_single_data(0, 100, (2), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)

        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_int32(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_bool_int32(self):  # wt
        npu_input1 = self.generate_single_data(0, 10, (2, 2), np.int32)
        npu_input2 = self.generate_single_data(5, 10, (2, 2), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2 > 5)
        npu_output = self.npu_op_exec(npu_input1, npu_input2 > 5)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_tensor_bool_int32(self):  # wt
        npu_input1 = self.generate_single_data(0, 10, (2, 2), np.int32)
        npu_input3 = self.generate_single_data(5, 10, (2, 2), np.int32)
        cpu_output1 = self.cpu_op_exec(npu_input1 > 5, npu_input3 > 5)
        npu_output1 = self.npu_op_exec(npu_input1 > 5, npu_input3 > 5)
        cpu_output2 = self.cpu_op_exec(npu_input1 > 5, 1.2)
        npu_output2 = self.npu_op_exec_scalar(npu_input1 > 5, 1.2)

        mask = ~(np.isnan(cpu_output1) | np.isinf(cpu_output1))
        self.assertRtolEqual(cpu_output1[mask], npu_output1[mask])

        mask = ~(np.isnan(cpu_output2) | np.isinf(cpu_output2))
        self.assertRtolEqual(cpu_output2[mask], npu_output2[mask])

    def test_true_divide_bool_scalar_int32(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 2), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, True)
        npu_output = self.npu_op_exec_scalar(npu_input1, True)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_scalar_int32_1_int32(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, 2)
        npu_output = self.npu_op_exec_scalar(npu_input1, 2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_scalar_float32(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, 2.0)
        npu_output = self.npu_op_exec_scalar(npu_input1, 2.0)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    # out
    def cpu_op_out_exec(self, input1, input2):
        c = torch.randn(5, 4, 4).uniform_(-100, 100).to(torch.float32)
        output = torch.true_divide(input1, input2, out=c)
        output = output.numpy()
        return output

    def npu_op_out_exec(self, input1, input2):
        c = torch.randn(5, 3).uniform_(-100, 100).to(torch.float32)
        nc = c.npu()
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.true_divide(input1, input2, out=nc)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar_out(self, input1, input2):
        c = torch.randn(5, 3, 2).uniform_(-100, 100).to(torch.float32)
        nc = c.npu()
        input1 = input1.to("npu")
        output = torch.true_divide(input1, input2, out=nc)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_true_divide_int32_broadcast_out(self):
        npu_input1 = self.generate_single_data(0, 100, (2, 2), np.int32)
        npu_input2 = self.generate_single_data(0, 100, (2), np.int32)
        cpu_output = self.cpu_op_out_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_out_exec(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_float32_broadcast_out(self):
        npu_input1 = self.generate_single_data(0, 100, (2, 2), np.float32)
        npu_input2 = self.generate_single_data(0, 100, (2), np.float32)
        cpu_output = self.cpu_op_out_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_out_exec(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_float16_broadcast_out(self):  # wt
        npu_input1 = self.generate_single_data(0, 100, (2, 2), np.float16)
        npu_input2 = self.generate_single_data(0, 100, (2), np.float16)
        cpu_output = self.cpu_op_out_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_out_exec(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_int32_out(self):  # wt
        npu_input1, npu_input2 = self.generate_data(0, 100, (5, 3, 2, 4), np.int32)
        cpu_output = self.cpu_op_out_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_out_exec(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_float32_out(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (5, 3, 2, 4), np.float32)
        cpu_output = self.cpu_op_out_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_out_exec(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_float16_out(self):  # wt
        npu_input1, npu_input2 = self.generate_data(0, 100, (5, 3, 2, 4), np.float16)
        cpu_output = self.cpu_op_out_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_out_exec(npu_input1, npu_input2)

        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_bool_int32_out(self):
        npu_input1 = self.generate_single_data(0, 10, (2, 2), np.int32)
        npu_input2 = self.generate_single_data(5, 10, (2, 2), np.int32)
        cpu_output = self.cpu_op_out_exec(npu_input1, npu_input2 > 5)
        npu_output = self.npu_op_out_exec(npu_input1, npu_input2 > 5)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_bool_float32_out(self):
        npu_input1 = self.generate_single_data(0, 10, (2, 2), np.float32)
        npu_input2 = self.generate_single_data(5, 10, (2, 2), np.float32)
        cpu_output = self.cpu_op_out_exec(npu_input1, npu_input2 > 5)
        npu_output = self.npu_op_out_exec(npu_input1, npu_input2 > 5)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_bool_float16_out(self):
        npu_input1 = self.generate_single_data(0, 10, (2, 2), np.float16)
        npu_input2 = self.generate_single_data(5, 10, (2, 2), np.float16)
        cpu_output = self.cpu_op_out_exec(npu_input1, npu_input2 > 5)
        npu_output = self.npu_op_out_exec(npu_input1, npu_input2 > 5)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_tensor_bool_int32_out(self):  # wt
        npu_input1 = self.generate_single_data(0, 10, (2, 2), np.int32)
        npu_input3 = self.generate_single_data(5, 10, (2, 2), np.int32)
        cpu_output1 = self.cpu_op_out_exec(npu_input1 > 5, npu_input3 > 5)
        npu_output1 = self.npu_op_out_exec(npu_input1 > 5, npu_input3 > 5)
        mask = ~(np.isnan(cpu_output1) | np.isinf(cpu_output1))
        self.assertRtolEqual(cpu_output1[mask], npu_output1[mask])

        cpu_output2 = self.cpu_op_out_exec(npu_input1 > 5, 1.2)
        npu_output2 = self.npu_op_exec_scalar_out(npu_input1 > 5, 1.2)
        mask = ~(np.isnan(cpu_output2) | np.isinf(cpu_output2))
        self.assertRtolEqual(cpu_output2[mask], npu_output2[mask])

    def test_true_divide_tensor_bool_float32_out(self):
        npu_input1 = self.generate_single_data(0, 10, (2, 2), np.float32)
        npu_input3 = self.generate_single_data(5, 10, (2, 2), np.float32)
        cpu_output1 = self.cpu_op_out_exec(npu_input1 > 5, npu_input3 > 5)
        npu_output1 = self.npu_op_out_exec(npu_input1 > 5, npu_input3 > 5)
        mask = ~(np.isnan(cpu_output1) | np.isinf(cpu_output1))
        self.assertRtolEqual(cpu_output1[mask], npu_output1[mask])

        cpu_output2 = self.cpu_op_out_exec(npu_input1 > 5, 1.2)
        npu_output2 = self.npu_op_exec_scalar_out(npu_input1 > 5, 1.2)
        mask = ~(np.isnan(cpu_output2) | np.isinf(cpu_output2))
        self.assertRtolEqual(cpu_output2[mask], npu_output2[mask])

    def test_true_divide_tensor_bool_float16_out(self):
        npu_input1 = self.generate_single_data(0, 10, (2, 2), np.float16)
        npu_input3 = self.generate_single_data(5, 10, (2, 2), np.float16)
        cpu_output1 = self.cpu_op_out_exec(npu_input1 > 5, npu_input3 > 5)
        npu_output1 = self.npu_op_out_exec(npu_input1 > 5, npu_input3 > 5)
        mask = ~(np.isnan(cpu_output1) | np.isinf(cpu_output1))
        self.assertRtolEqual(cpu_output1[mask], npu_output1[mask], 0.001)

        cpu_output2 = self.cpu_op_out_exec(npu_input1 > 5, 1.2)
        npu_output2 = self.npu_op_exec_scalar_out(npu_input1 > 5, 1.2)
        mask = ~(np.isnan(cpu_output2) | np.isinf(cpu_output2))
        self.assertRtolEqual(cpu_output2[mask], npu_output2[mask], 0.001)

    def test_true_divide_bool_scalar_int32_out(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 2), np.int32)
        cpu_output = self.cpu_op_out_exec(npu_input1, True)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, True)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_bool_scalar_float32_out(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 2), np.float32)
        cpu_output = self.cpu_op_out_exec(npu_input1, True)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, True)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_bool_scalar_float16_out(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 2), np.float16)
        cpu_output = self.cpu_op_out_exec(npu_input1, True)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, True)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_scalar_int32_1_int32_out(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.int32)
        cpu_output = self.cpu_op_out_exec(npu_input1, 2)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, 2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_scalar_float32_1_int32_out(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.float32)
        cpu_output = self.cpu_op_out_exec(npu_input1, 2)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, 2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_scalar_float16_1_int32_out(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.float16)
        cpu_output = self.cpu_op_out_exec(npu_input1, 2)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, 2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_scalar_int32_out(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.int32)
        cpu_output = self.cpu_op_out_exec(npu_input1, 2.0)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, 2.0)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_scalar_float32_out(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.float32)
        cpu_output = self.cpu_op_out_exec(npu_input1, 2.0)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, 2.0)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_scalar_float16_out(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.float16)
        cpu_output = self.cpu_op_out_exec(npu_input1, 2.0)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, 2.0)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

        # yd-float

    def cpu_op_exec_inplace(self, input1, input2):
        input1.true_divide_(input2)
        input1 = input1.numpy()
        return input1

    def npu_op_exec_inplace(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input1.true_divide_(input2)
        input1 = input1.to("cpu")
        input1 = input1.numpy()
        return input1

    def npu_op_exec_scalar_inplace(self, input1, input2):
        input1 = input1.to("npu")
        input1.true_divide_(input2)
        input1 = input1.to("cpu")
        input1 = input1.numpy()
        return input1

    def generate_data_inplace(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        return npu_input1, npu_input2

    def generate_single_data_inplace(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def test_true_divide_float32_broadcast_inplace(self):
        item = [[np.float32, 0, (2, 2)]]
        item1 = [[np.float32, 0, (2,)]]

        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        cpu_input2, npu_input2 = create_common_tensor(item1[0], 0, 100)

        cpu_output = self.cpu_op_exec_inplace(cpu_input1, cpu_input2)
        npu_output = self.npu_op_exec_inplace(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_float16_broadcast_inplace(self):
        item = [[np.float16, 0, (2, 2)]]
        item1 = [[np.float16, 0, (2,)]]

        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        cpu_input2, npu_input2 = create_common_tensor(item1[0], 0, 100)

        cpu_output = self.cpu_op_exec_inplace(cpu_input1, cpu_input2)
        npu_output = self.npu_op_exec_inplace(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_float32_inplace(self):
        item = [[np.float32, 0, (5, 3, 2, 4)]]
        item1 = [[np.float32, 0, (5, 3, 2, 4)]]

        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        cpu_input2, npu_input2 = create_common_tensor(item1[0], 0, 100)

        cpu_output = self.cpu_op_exec_inplace(cpu_input1, cpu_input2)
        npu_output = self.npu_op_exec_inplace(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_float16_inplace(self):
        item = [[np.float16, 0, (5, 3, 2, 4)]]
        item1 = [[np.float16, 0, (5, 3, 2, 4)]]

        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        cpu_input2, npu_input2 = create_common_tensor(item1[0], 0, 100)

        cpu_output = self.cpu_op_exec_inplace(cpu_input1, cpu_input2)

        npu_output = self.npu_op_exec_inplace(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_bool_float32_inplace(self):  # wt
        item = [[np.float32, 0, (5, 3, 2, 4)]]
        item1 = [[np.float32, 0, (5, 3, 2, 4)]]

        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        cpu_input2, npu_input2 = create_common_tensor(item1[0], 0, 100)

        cpu_output = self.cpu_op_exec_inplace(cpu_input1, cpu_input2 > 5)
        npu_output = self.npu_op_exec_inplace(npu_input1, npu_input2 > 5)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_bool_float16_inplace(self):  # wt
        item = [[np.float16, 0, (5, 3, 2, 4)]]
        item1 = [[np.float16, 0, (5, 3, 2, 4)]]

        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        cpu_input2, npu_input2 = create_common_tensor(item1[0], 0, 100)

        cpu_output = self.cpu_op_exec_inplace(cpu_input1, cpu_input2 > 5)
        npu_output = self.npu_op_exec_inplace(npu_input1, npu_input2 > 5)

        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_bool_scalar_float32_inplace(self):
        item = [[np.float32, 0, (5, 3, 2, 4)]]
        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        cpu_output = self.cpu_op_exec_inplace(cpu_input1, True)
        npu_output = self.npu_op_exec_scalar_inplace(npu_input1, True)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_bool_scalar_float16_inplace(self):
        item = [[np.float16, 0, (5, 3, 2, 4)]]
        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        cpu_output = self.cpu_op_exec_inplace(cpu_input1, True)
        npu_output = self.npu_op_exec_scalar_inplace(npu_input1, True)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_scalar_float32_1_int32_inplace(self):
        item = [[np.float32, 0, (5, 3, 2, 4)]]
        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        cpu_output = self.cpu_op_exec_inplace(cpu_input1, 2)
        npu_output = self.npu_op_exec_scalar_inplace(npu_input1, 2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_scalar_float16_1_int32_inplace(self):
        item = [[np.float16, 0, (5, 3, 2, 4)]]
        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        cpu_output = self.cpu_op_exec_inplace(cpu_input1, 2)
        npu_output = self.npu_op_exec_scalar_inplace(npu_input1, 2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_scalar_float32_inplace(self):
        item = [[np.float32, 0, (5, 3, 2, 4)]]
        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        cpu_output = self.cpu_op_exec_inplace(cpu_input1, 2.0)
        npu_output = self.npu_op_exec_scalar_inplace(npu_input1, 2.0)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_scalar_float16_inplace(self):
        item = [[np.float16, 0, (5, 3, 2, 4)]]
        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        cpu_output = self.cpu_op_exec_inplace(cpu_input1, 2.0)
        npu_output = self.npu_op_exec_scalar_inplace(npu_input1, 2.0)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    # pt-float
    def test_true_divide_float32_broadcast_float(self):
        npu_input1 = self.generate_single_data(0, 100, (2, 2), np.float32)
        npu_input2 = self.generate_single_data(0, 100, (2), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_float16_broadcast_float(self):
        npu_input1 = self.generate_single_data(0, 100, (2, 2), np.float16)
        npu_input2 = self.generate_single_data(0, 100, (2), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_float32_float(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_float16_float(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_bool_float(self):
        npu_input1 = self.generate_single_data(0, 10, (2, 2), np.float32)
        npu_input2 = self.generate_single_data(5, 10, (2, 2), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2 > 5)
        npu_output = self.npu_op_exec(npu_input1, npu_input2 > 5)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_bool_float16(self):
        npu_input1 = self.generate_single_data(0, 10, (2, 2), np.float16)
        npu_input2 = self.generate_single_data(5, 10, (2, 2), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2 > 5)
        npu_output = self.npu_op_exec(npu_input1, npu_input2 > 5)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_tensor_bool_float(self):
        npu_input1 = self.generate_single_data(0, 10, (2, 2), np.float32)
        npu_input3 = self.generate_single_data(5, 10, (2, 2), np.float32)
        cpu_output1 = self.cpu_op_exec(npu_input1 > 5, npu_input3 > 5)
        npu_output1 = self.npu_op_exec(npu_input1 > 5, npu_input3 > 5)
        cpu_output2 = self.cpu_op_exec(npu_input1 > 5, 1.2)
        npu_output2 = self.npu_op_exec_scalar(npu_input1 > 5, 1.2)
        mask = ~(np.isnan(cpu_output1) | np.isinf(cpu_output1))
        self.assertRtolEqual(cpu_output1[mask], npu_output1[mask])
        mask = ~(np.isnan(cpu_output2) | np.isinf(cpu_output2))
        self.assertRtolEqual(cpu_output2[mask], npu_output2[mask])

    def test_true_divide_tensor_bool_float16(self):
        npu_input1 = self.generate_single_data(0, 10, (2, 2), np.float16)
        npu_input3 = self.generate_single_data(5, 10, (2, 2), np.float16)
        cpu_output1 = self.cpu_op_exec(npu_input1 > 5, npu_input3 > 5)
        npu_output1 = self.npu_op_exec(npu_input1 > 5, npu_input3 > 5)
        cpu_output2 = self.cpu_op_exec(npu_input1 > 5, 1.2)
        npu_output2 = self.npu_op_exec_scalar(npu_input1 > 5, 1.2)
        mask = ~(np.isnan(cpu_output1) | np.isinf(cpu_output1))
        self.assertRtolEqual(cpu_output1[mask], npu_output1[mask], 0.001)
        mask = ~(np.isnan(cpu_output2) | np.isinf(cpu_output2))
        self.assertRtolEqual(cpu_output2[mask], npu_output2[mask], 0.001)

    def test_true_divide_bool_scalar_float(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 2), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, True)
        npu_output = self.npu_op_exec_scalar(npu_input1, True)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_bool_scalar_float16(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 2), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, True)
        npu_output = self.npu_op_exec_scalar(npu_input1, True)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_scalar_int32_1_float(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, 2)
        npu_output = self.npu_op_exec_scalar(npu_input1, 2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_scalar_int32_1_float16(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, 2)
        npu_output = self.npu_op_exec_scalar(npu_input1, 2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_scalar_int32_2_float(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, 2)
        npu_output = self.npu_op_exec_scalar(npu_input1, 2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_scalar_int32_2_float16(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, 2)
        npu_output = self.npu_op_exec_scalar(npu_input1, 2)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_true_divide_scalar_float32_float(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, 2.0)
        npu_output = self.npu_op_exec_scalar(npu_input1, 2.0)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def test_true_divide_scalar_float32_float16(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, 2.0)
        npu_output = self.npu_op_exec_scalar(npu_input1, 2.0)
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)


if __name__ == "__main__":
    run_tests()
