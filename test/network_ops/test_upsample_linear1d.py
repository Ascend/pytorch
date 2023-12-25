import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestUpsampleLinear1D(TestCase):
    def cpu_op_exec(self, input1, size, align_corners):
        out_result = torch.ones(input1.shape[0], input1.shape[1], size[0], dtype=input1.dtype)
        output = torch._C._nn.upsample_linear1d(input=input1, output_size=size, align_corners=align_corners)
        torch._C._nn.upsample_linear1d(input=input1, output_size=size, align_corners=align_corners, out=out_result)
        return output.numpy(), out_result.numpy()

    def npu_op_exec(self, input1, size, align_corners):
        out_result = torch.ones(input1.shape[0], input1.shape[1], size[0], dtype=input1.dtype)
        out_result = out_result.to("npu")
        output = torch._C._nn.upsample_linear1d(input=input1, output_size=size, align_corners=align_corners)
        torch._C._nn.upsample_linear1d(input=input1, output_size=size, align_corners=align_corners, out=out_result)
        output = output.to("cpu")
        out_result = out_result.to("cpu")
        return output.numpy(), out_result.numpy()

    def cpu_op_scale_exec(self, input1, size):
        output = torch.nn.functional.interpolate(input1, scale_factor=size, mode="linear")
        output = output.numpy()
        return output

    def npu_op_scale_exec(self, input1, size):
        output = torch.nn.functional.interpolate(input1, scale_factor=size, mode="linear")
        output = output.to("cpu")
        output = output.numpy()
        return output

    def creat_shape_format1(self):
        test_cases = [
            [[np.float16, 0, (1, 1, 1, 2)], [4, ], True],
            [[np.float16, 0, (2, 1, 1, 4)], [8, ], True],
            [[np.float16, 0, (2, 2, 1, 3)], [1, ], True],
            [[np.float16, 0, (2, 1, 1, 1)], [4, ], False],
            [[np.float16, 0, (4, 1, 1, 2)], [4, ], False],
            [[np.float16, 0, (1, 1, 1, 1)], [1, ], False],

            [[np.float32, 0, (1, 1, 1, 2)], [4, ], True],
            [[np.float32, 0, (2, 1, 1, 2)], [4, ], True],
            [[np.float32, 0, (2, 2, 1, 3)], [1, ], True],
            [[np.float32, 0, (3, 1, 1, 1)], [2, ], False],
            [[np.float32, 0, (4, 1, 1, 1)], [2, ], False],
            [[np.float32, 0, (1, 1, 1, 1)], [1, ], False],

            [[np.float16, 0, (9, 7, 1, 2)], [15, ], True],
            [[np.float16, 0, (8, 7, 1, 1)], [2, ], True],
            [[np.float16, 0, (17, 2, 1, 3)], [1, ], True],
            [[np.float16, 0, (6, 4, 1, 1)], [3, ], False],
            [[np.float16, 0, (8, 7, 1, 2)], [4, ], False],
            [[np.float16, 0, (2, 7, 1, 7)], [1, ], False],

            [[np.float32, 0, (9, 7, 1, 2)], [7, ], True],
            [[np.float32, 0, (8, 3, 1, 1)], [2, ], True],
            [[np.float32, 0, (17, 2, 1, 3)], [1, ], True],
            [[np.float32, 0, (9, 7, 1, 2)], [7, ], False],
            [[np.float32, 0, (8, 3, 1, 3)], [2, ], False],
            [[np.float32, 0, (2, 7, 1, 7)], [1, ], False],

            [[np.float16, 0, (9, 7, 1, 2)], [17, ], True],
            [[np.float16, 0, (17, 13, 1, 15)], [16, ], True],
            [[np.float16, 0, (61, 41, 1, 1)], [7, ], False],
            [[np.float16, 0, (38, 7, 1, 7)], [16, ], False],
            [[np.float32, 0, (997, 3, 1, 1)], [32, ], True],
            [[np.float32, 0, (627, 2, 1, 3)], [17, ], False],
            [[np.float32, 0, (78, 73, 1, 1)], [48, ], False],
            [[np.float32, 0, (65535, 2, 1, 4)], [8, ], False],
            [[np.float16, 0, (65535, 2, 1, 4)], [8, ], False],
            [[np.float32, 0, (10086, 3, 1, 17)], [57, ], False],
            [[np.float16, 0, (10086, 3, 1, 17)], [57, ], False]
        ]
        return test_cases

    def test_upsample_linear1d(self):
        for item in self.creat_shape_format1():
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            if cpu_input.dim() == 4:
                cpu_input = cpu_input.squeeze(2)

            if npu_input.dim() == 4:
                npu_input = npu_input.squeeze(2)

            size = item[1]
            align_corners = item[2]

            npu_output, npu_out_result = self.npu_op_exec(npu_input, size, align_corners)
            cpu_output, cpu_out_result = self.cpu_op_exec(cpu_input, size, align_corners)

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_out_result = cpu_out_result.astype(npu_out_result.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_out_result, npu_out_result)

    def test_upsample_scale_linear1d(self):
        for item in self.creat_shape_format1():
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            if cpu_input.dim() == 4:
                cpu_input = cpu_input.squeeze(2)

            if npu_input.dim() == 4:
                npu_input = npu_input.squeeze(2)

            size = item[1]

            npu_output = self.npu_op_scale_exec(npu_input, size)
            cpu_output = self.cpu_op_scale_exec(cpu_input, size)

            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
