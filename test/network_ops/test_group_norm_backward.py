import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGroupNormBackward(TestCase):
    def cpu_op_exec(self, input1, num_groups):
        input1.requires_grad = True
        output = torch.nn.functional.group_norm(input1, num_groups)
        output.backward(torch.ones_like(output))
        input_grad = input1.grad

        output = output.detach().numpy()
        input_grad = input_grad.detach().numpy()

        return output, input_grad

    def cpu_op_fp16_exec(self, input1, num_groups):
        input1 = input1.to(torch.float32)
        input1.requires_grad = True
        output = torch.nn.functional.group_norm(input1, num_groups)
        output.backward(torch.ones_like(output))
        input_grad = input1.grad

        output = output.detach().numpy().astype(np.float16)
        input_grad = input_grad.detach().numpy().astype(np.float16)

        return output, input_grad

    def npu_op_exec(self, input1, num_groups):
        input1.requires_grad = True
        output = torch.nn.functional.group_norm(input1, num_groups).to("npu")
        output.backward(torch.ones_like(output))
        input_grad = input1.grad

        output = output.detach().cpu().numpy()
        input_grad = input_grad.detach().cpu().numpy()

        return output, input_grad

    def test_GroupNorm_Backward_default_fp32(self):
        # create inputs and params
        shape_format = [
            [[np.float32, 0, [20, 6, 10, 10]], 2],
            [[np.float32, 3, [20, 6, 10, 10]], 2],
            [[np.float32, 0, [20, 2, 10, 10]], 2],
            [[np.float32, 3, [20, 2, 10, 10]], 2],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)

            # run exec
            cpu_output, cpu_input_grad = self.cpu_op_exec(cpu_input1, item[1])
            npu_output, npu_input_grad = self.npu_op_exec(npu_input1, item[1])

            # results comparison
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_GroupNorm_Backward_default_fp16(self):
        # create inputs and params
        shape_format = [
            [[np.float16, 0, [20, 6, 10, 10]], 2],
            [[np.float16, 3, [20, 6, 10, 10]], 2],
            [[np.float16, 0, [20, 2, 10, 10]], 2],
            [[np.float16, 3, [20, 2, 10, 10]], 2],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)

            # run exec
            cpu_output, cpu_input_grad = self.cpu_op_fp16_exec(cpu_input1, item[1])
            npu_output, npu_input_grad = self.npu_op_exec(npu_input1, item[1])

            # results comparison
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_GroupNorm_Backward_case1(self):
        # create inputs and params
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

            # run exec
            cpu_output, cpu_input_grad = self.cpu_op_exec(cpu_input1, item[1])
            npu_output, npu_input_grad = self.npu_op_exec(npu_input1, item[1])

            # results comparison
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_GroupNorm_Backward_case2(self):
        # create inputs and params
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

            # run exec
            cpu_output, cpu_input_grad = self.cpu_op_exec(cpu_input1, item[1])
            npu_output, npu_input_grad = self.npu_op_exec(npu_input1, item[1])

            # results comparison
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_GroupNorm_Backward_case3(self):
        # create inputs and params
        shape_format = [
            [[np.float32, 0, [1, 256, 100, 152]], 32],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)

            # run exec
            cpu_output, cpu_input_grad = self.cpu_op_exec(cpu_input1, item[1])
            npu_output, npu_input_grad = self.npu_op_exec(npu_input1, item[1])

            # results comparison
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)


if __name__ == "__main__":
    run_tests()
