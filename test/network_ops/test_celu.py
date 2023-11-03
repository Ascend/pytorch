import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestCelu(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        # modify from numpy.ndarray to torch.tensor
        npu_input = torch.from_numpy(input_x)
        return npu_input

    def cpu_op_exec_functional(self, input1, alpha):
        output = torch.nn.functional.celu(input1, alpha=alpha)
        output = output.numpy()
        return output

    def npu_op_exec_functional(self, input1, alpha):
        output = torch.nn.functional.celu(input1, alpha=alpha)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec(self, input1, alpha):
        output = torch.celu(input1, alpha=alpha)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, alpha):
        output = torch.celu(input1, alpha=alpha)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec_functional(self, input1, alpha):
        output = torch.nn.functional.celu_(input1, alpha=alpha)
        output = output.numpy()
        return output

    def npu_op_inplace_exec_functional(self, input1, alpha):
        output = torch.nn.functional.celu_(input1, alpha=alpha)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec(self, input1, alpha):
        output = torch.celu_(input1, alpha=alpha)
        output = output.numpy()
        return output

    def npu_op_inplace_exec(self, input1, alpha):
        output = torch.celu_(input1, alpha=alpha)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_fp16(self, input1, alpha):
        input1 = input1.to(torch.float32)
        output = torch.nn.functional.celu(input1, alpha=alpha)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def test_celu_3_3_float32_alpha1(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 1.0)
        npu_output1 = self.npu_op_exec(input_x1, 1.0)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_celu_10_10_10_10_float32_alpha1(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (10, 10, 10, 10), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 1.0)
        npu_output1 = self.npu_op_exec(input_x1, 1.0)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_celu_100_100_float32_alpha2(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (100, 100), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 2.0)
        npu_output1 = self.npu_op_exec(input_x1, 2.0)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_celu_float16_alpha1(self, device="npu"):
        shape_format = [
            [[np.float16, 0, (65535, 1, 1, 1)]],
            [[np.float16, 0, (1, 1, 1, 65535)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            cpu_output = self.cpu_op_exec_fp16(cpu_input1, 1.0)
            npu_output = self.npu_op_exec(npu_input1, 1.0)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_celu_float16_alpha2_success(self, device="npu"):
        shape_format = [
            [[np.float16, 0, (65535, 1, 1, 1)]],
            [[np.float16, 0, (1, 1, 1, 65535)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec_fp16(cpu_input1, 2.0)
            npu_output = self.npu_op_exec(npu_input1, 2.0)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_celu_float16_alpha2_fail(self, device="npu"):
        shape_format = [
            [[np.float16, 0, (65535, 1, 1, 1)]],
            [[np.float16, 0, (1, 1, 1, 65535)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            cpu_output = self.cpu_op_exec_fp16(cpu_input1, 2.0)
            npu_output = self.npu_op_exec(npu_input1, 2.0)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_celu_inplace_alpha1(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (65535, 1, 1, 1)]],
            [[np.float32, 0, (1, 1, 1, 65535)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            cpu_output = self.cpu_op_inplace_exec(cpu_input1, 1.0)
            npu_output = self.npu_op_inplace_exec(npu_input1, 1.0)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_celu_inplace_alpha2(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (65535, 1, 1, 1)]],
            [[np.float32, 0, (1, 1, 1, 65535)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_inplace_exec(cpu_input1, 2.0)
            npu_output = self.npu_op_inplace_exec(npu_input1, 2.0)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_celu_inplace_alpha2_fail(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (65535, 1, 1, 1)]],
            [[np.float32, 0, (1, 1, 1, 65535)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            cpu_output = self.cpu_op_inplace_exec(cpu_input1, 2.0)
            npu_output = self.npu_op_inplace_exec(npu_input1, 2.0)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_celu_inplace_shape_format_alpha_range(self, device="npu"):
        shape_format_alpha_range = [
            # 注：[[dtype, format, shape], alpha, min, max]
            [[np.float16, 2, (16, 5, 7, 11)], 5.6, -2, 2],
            [[np.float32, 2, (16, 5, 7, 11)], 0.5, -2, 2],
            [[np.float32, 2, (16, 5, 7, 11)], 0.7, -2, 2],
            [[np.float32, 2, (16, 5, 7, 11)], 2.6, -2, 2],
            [[np.float16, 2, (16, 136, 5, 4)], 0.5, -0.0078125, 0.0078125],
            [[np.float16, 2, (16, 136, 5, 4)], 0.7, -0.0078125, 0.0078125],
            [[np.float16, 2, (16, 136, 5, 4)], 0.5, -0.01, 0.01],
            [[np.float16, 2, (176, 3, 67, 47, 5, 12)], 0.5, -2, 2],
            [[np.float16, 2, (176, 3, 67, 47, 5, 12)], 5.4, -2, 2],
            [[np.float16, 2, (23, 5, 11, 50, 26, 13, 1, 23)], 0.5, -2, 2],
            [[np.float16, 2, (2560, 17)], 0.5, -2, 2],
            [[np.float16, 2, (2560, 17)], 5.4, -2, 2]
        ]
        for item in shape_format_alpha_range:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[2], item[3])
            alpha = item[1]
            npu_output = self.npu_op_inplace_exec(npu_input1, alpha)
            if item[0][0] == np.float16:
                cpu_output = self.cpu_op_inplace_exec(cpu_input1.float(), alpha).astype(np.float16)
            else:
                cpu_output = self.cpu_op_inplace_exec(cpu_input1, alpha)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_celu_inplace_shape_format_alpha_range(self, device="npu"):
        shape_format_alpha_range = [
            # 注：[[dtype, format, shape], alpha, min, max]
            [[np.float32, 2, (16, 5, 7, 11)], 0.5, -2, 2],
            [[np.float32, 2, (16, 5, 7, 11)], 0.7, -2, 2],
            [[np.float32, 2, (16, 5, 7, 11)], 2.6, -2, 2],
            [[np.float16, 2, (16, 136, 5, 4)], 0.5, -0.0078125, 0.0078125],
            [[np.float16, 2, (16, 136, 5, 4)], 0.7, -0.0078125, 0.0078125],
            [[np.float16, 2, (16, 136, 5, 4)], 0.5, -0.01, 0.01],
            [[np.float16, 2, (16, 136, 5, 4)], 0.7, -0.01, 0.01],
            [[np.float16, 2, (176, 3, 67, 47, 5, 12)], 0.5, -2, 2],
            [[np.float16, 2, (176, 3, 67, 47, 5, 12)], 5.4, -2, 2],
            [[np.float16, 2, (2560, 17)], 0.5, -2, 2],
            [[np.float16, 2, (2560, 17)], 5.4, -2, 2]
        ]
        for item in shape_format_alpha_range:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[2], item[3])
            alpha = item[1]
            npu_output = self.npu_op_exec(npu_input1, alpha)
            if item[0][0] == np.float16:
                cpu_output = self.cpu_op_exec(cpu_input1.float(), alpha).astype(np.float16)
            else:
                cpu_output = self.cpu_op_exec(cpu_input1, alpha)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
