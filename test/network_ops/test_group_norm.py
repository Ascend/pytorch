import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGroupNorm(TestCase):
    def cpu_op_exec(self, input1, num_groups):
        output = torch.nn.functional.group_norm(input1, num_groups)
        output_cpu = output.detach().numpy()
        return output_cpu

    def cpu_op_fp16_exec(self, input1, num_groups):
        input1 = input1.to(torch.float32)
        output = torch.nn.functional.group_norm(input1, num_groups)
        output_cpu = output.numpy().astype(np.float16)
        return output_cpu

    def npu_op_exec(self, input1, num_groups):
        output = torch.nn.functional.group_norm(input1, num_groups).to("npu")
        output = output.to("cpu").detach().numpy()
        return output

    def test_GroupNorm_default_fp32(self):
        shape_format = [
            [[np.float32, 0, [20, 6, 10, 10]], 2],
            [[np.float32, 3, [20, 6, 10, 10]], 2],
            [[np.float32, 0, [20, 2, 10, 10]], 2],
            [[np.float32, 3, [20, 2, 10, 10]], 2],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_GroupNorm_default_fp16(self):
        shape_format = [
            [[np.float16, 0, [20, 6, 10, 10]], 2],
            [[np.float16, 3, [20, 6, 10, 10]], 2],
            [[np.float16, 0, [20, 2, 10, 10]], 2],
            [[np.float16, 3, [20, 2, 10, 10]], 2],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_fp16_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_GroupNorm_case1(self):
        shape_format = [
            [[np.float32, 0, [48, 32, 320, 320]], 32],
            [[np.float32, 0, [48, 64, 160, 160]], 32],
            [[np.float32, 0, [48, 32, 160, 160]], 32],
            [[np.float32, 0, [48, 128, 80, 80]], 32],
            [[np.float32, 0, [48, 64, 80, 80]], 32],
            [[np.float32, 0, [48, 256, 40, 40]], 32],
            [[np.float32, 0, [48, 128, 40, 40]], 32],
            [[np.float32, 0, [48, 512, 20, 20]], 32],
            [[np.float32, 0, [48, 256, 20, 20]], 32],
            [[np.float32, 0, [48, 512, 20, 20]], 32],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_GroupNorm_case2(self):
        shape_format = [
            [[np.float32, 0, [4, 128, 384, 384]], 32],
            [[np.float32, 0, [4, 128, 768, 768]], 32],
            [[np.float32, 0, [4, 1280, 12, 12]], 32],
            [[np.float32, 0, [4, 1280, 24, 24]], 32],
            [[np.float32, 0, [4, 1280, 48, 48]], 32],
            [[np.float32, 0, [4, 1920, 24, 24]], 32],
            [[np.float32, 0, [4, 1920, 48, 48]], 32],
            [[np.float32, 0, [4, 256, 192, 192]], 32],
            [[np.float32, 0, [4, 256, 384, 384]], 32],
            [[np.float32, 0, [4, 2560, 12, 12]], 32],
            [[np.float32, 0, [4, 2560, 24, 24]], 32],
            [[np.float32, 0, [4, 320, 48, 48]], 32],
            [[np.float32, 0, [4, 320, 96, 96]], 32],
            [[np.float32, 0, [4, 512, 192, 96]], 32],
            [[np.float32, 0, [4, 512, 9216]], 32],
            [[np.float32, 0, [4, 512, 96, 96]], 32],
            [[np.float32, 0, [4, 640, 24, 24]], 32],
            [[np.float32, 0, [4, 640, 48, 48]], 32],
            [[np.float32, 0, [4, 640, 96, 96]], 32],
            [[np.float32, 0, [4, 960, 48, 48]], 32],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_GroupNorm_case3(self):
        shape_format = [
            [[np.float32, 0, [1, 256, 100, 152]], 32],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
