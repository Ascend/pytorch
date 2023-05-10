import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestReflectionPad2dBackward(TestCase):

    def cpu_op_exec(self, input1, pad):
        m = torch.nn.ReflectionPad2d(pad)
        input1.requires_grad = True
        output = m(input1)
        output.backward(torch.ones_like(output))
        input_grad = input1.grad
        output = output.numpy()
        input_grad = input_grad.numpy()
        return output, input_grad

    def npu_op_exec(self, input1, pad):
        m = torch.nn.ReflectionPad2d(pad).to("npu")
        input1.requires_grad = True
        output = m(input1)
        output.backward(torch.ones_like(output))
        input_grad = input1.grad
        output = output.to("cpu")
        output = output.detach().numpy()
        input_grad = input_grad.cpu().numpy()
        return output, input_grad

    def test_reflectionPad2d_backward_shape_format_fp16(self):
        shape_format = [
            [[np.float16, 0, (1, 1, 37, 37)], [2, 2, 2, 2]],
            [[np.float16, 3, (1, 1, 4, 3)], 2],
            [[np.float16, 0, (1, 1, 17, 17)], [1, 2, 2, 2]],
            [[np.float16, 0, (2, 4, 3)], 2],
            [[np.float16, 0, (4, 17, 17)], [1, 2, 2, 2]]
        ]

        def cpu_op_exec_fp16(input1, pad):
            input1 = input1.to(torch.float32)
            input1.requires_grad = True
            m = torch.nn.ReflectionPad2d(pad)
            output = m(input1)            
            output.backward(torch.ones_like(output))
            output = output.detach().numpy()
            input_grad = input1.grad
            input_grad = input_grad.numpy().astype(np.float16)
            output = output.astype(np.float16)
            return output, input_grad
        
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output, cpu_grad = cpu_op_exec_fp16(cpu_input1, item[1])
            npu_output, npu_grad = self.npu_op_exec(npu_input1, item[1])     
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
