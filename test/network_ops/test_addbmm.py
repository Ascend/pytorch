import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAddbmm(TestCase):
    def generate_scalar(self, dtype, min_d, max_d):
        if dtype == "float32":
            scalar = np.random.uniform(min_d, max_d)
        if dtype == "int32":
            scalar = np.random.randint(min_d, max_d)
        return scalar

    def cpu_op_exec(self, input1, input2, input3, scalar1, scalar2):
        output = torch.addbmm(input1, input2, input3, beta=scalar1, alpha=scalar2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, input3, scalar1, scalar2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = torch.addbmm(input1, input2, input3, beta=scalar1, alpha=scalar2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3, scalar1, scalar2, input4):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = input4.to("npu")
        torch.addbmm(input1, input2, input3, beta=scalar1, alpha=scalar2, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_inplace(self, input1, input2, input3, scalar1, scalar2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        input1.addbmm_(input2, input3, beta=scalar1, alpha=scalar2)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_transpose_exec(self, input1, input2, input3, scalar1, scalar2):
        input3_t = np.transpose(input3, (0, 2, 1))
        output = torch.addbmm(input1, input2, input3_t, beta=scalar1, alpha=scalar2)
        output = output.numpy()
        return output

    def npu_op_transpose_exec(self, input1, input2, input3, scalar1, scalar2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        input3_t = torch.permute(input3, (0, 2, 1))
        output = torch.addbmm(input1, input2, input3_t, beta=scalar1, alpha=scalar2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_addbmm(self):
        shape_format = [
            [[np.float16, 0, [3, 5]], [np.float16, 0, [10, 3, 4]], [np.float16, 0, [10, 4, 5]], "float32"],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 1)
            cpu_input4, npu_input4 = create_common_tensor(item[0], 0, 1)

            scalar1 = self.generate_scalar(item[3], 0, 2)
            scalar2 = self.generate_scalar(item[3], 0, 2)

            cpu_output = self.cpu_op_exec(cpu_input1.float(), cpu_input2.float(), cpu_input3.float(), scalar1, scalar2)
            npu_output = self.npu_op_exec(npu_input1.float(), npu_input2.float(), npu_input3.float(), scalar1, scalar2)

            npu_output1 = self.npu_op_exec_out(npu_input1.float(), npu_input2.float(), npu_input3.float(), scalar1, scalar2, npu_input4.float())
            npu_output2 = self.npu_op_exec_inplace(npu_input1.float(), npu_input2.float(), npu_input3.float(), scalar1, scalar2)

            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3, prec16=1.e-3)
            self.assertRtolEqual(cpu_output, npu_output1, prec=1.e-3, prec16=1.e-3)
            self.assertRtolEqual(cpu_output, npu_output2, prec=1.e-3, prec16=1.e-3)

    def test_addbmm_transpose(self):
        shape_format = [
            [[np.float16, 0, [4, 5]], [np.float16, 0, [10, 4, 7]], [np.float16, 0, [10, 5, 7]], "float32"],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 1)

            scalar1 = self.generate_scalar(item[3], 0, 2)
            scalar2 = self.generate_scalar(item[3], 0, 2)

            cpu_transpose_output = self.cpu_op_transpose_exec(cpu_input1.float(), cpu_input2.float(), cpu_input3.float(), scalar1, scalar2)
            npu_transpose_output = self.npu_op_transpose_exec(npu_input1.float(), npu_input2.float(), npu_input3.float(), scalar1, scalar2)

            self.assertRtolEqual(cpu_transpose_output, npu_transpose_output, prec=1.e-3, prec16=1.e-3)


if __name__ == "__main__":
    run_tests()
