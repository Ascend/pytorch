import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestConfusionTransposeDBackward(TestCase):
    def npu_op_exec(self, input1, shape, perm, transpose_first):
        input1.requires_grad_()
        output = torch_npu.npu_confusion_transpose(input1, perm, shape, transpose_first)
        output.backward(torch.ones_like(output))
        output1 = output.detach().cpu().numpy()
        output2 = input1.grad.cpu().numpy()
        return output1, output2

    def cpu_op_exec(self, input1, shape, perm, transpose_first):
        input1.requires_grad_()
        if transpose_first:
            output = input1.permute(*perm).contiguous().view(shape)
        else:
            output = input1.view(shape).permute(*perm)
        output.backward(torch.ones_like(output))
        output1 = output.detach().numpy()
        output2 = input1.grad.numpy()
        return output1, output2

    def test_confusion_transpose_backward(self, device="npu"):
        shape_format = [
            [[np.float32, 0, [1, 576, 2560]], [1, 576, 32, 80], (0, 2, 1, 3), False],
            [[np.float32, 0, [1, 32, 576, 80]], [1, 576, 2560], (0, 2, 1, 3), True],
            [[np.float16, 0, [1, 576, 2560]], [1, 576, 32, 80], (0, 2, 1, 3), False],
            [[np.float16, 0, [1, 32, 576, 80]], [1, 576, 2560], (0, 2, 1, 3), True],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output1, cpu_output2 = self.cpu_op_exec(cpu_input, item[1], item[2], item[3])
            npu_output1, npu_output2 = self.npu_op_exec(npu_input, item[1], item[2], item[3])
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)


if __name__ == "__main__":
    run_tests()
