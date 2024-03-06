import torch
import numpy as np
from torch.nn import functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestDropOutBackward(TestCase):
    def cpu_op_exec(self, input1):
        input1.requires_grad = True
        out = torch.nn.Dropout(0.5)(input1)
        out.backward(torch.ones_like(out))
        out_grad = input1.grad
        out_grad = out_grad.detach().numpy()
        out = out.detach().numpy()
        return out_grad, out

    def npu_op_exec(self, input1):
        input1.requires_grad = True
        out = torch.nn.Dropout(0.5)(input1)
        out.backward(torch.ones_like(out))
        out_grad = input1.grad
        out_grad = out_grad.to("cpu")
        out_grad = out_grad.detach().numpy()
        out = out.to("cpu")
        out = out.detach().numpy()
        return out_grad, out

    def dropout_list_exec(self, list1):
        epsilon = 1e-3
        for item in list1:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_grad, cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output_grad, npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            # 该算子随机结果的比较方式
            for a, b in zip(cpu_output.flatten(), npu_output.flatten()):
                if abs(a) > 0 and abs(b) > 0 and abs(a - b) > epsilon:
                    print(f'input = {item}, ERROR!')
                    break
            else:
                print(f'input = {item}, Successfully!')

            for a, b in zip(cpu_output_grad.flatten(), npu_output_grad.flatten()):
                if abs(a) > 0 and abs(b) > 0 and abs(a - b) > epsilon:
                    print(f'input = {item}, ERROR!')
                    break
            else:
                print(f'input = {item}, Successfully!')

    def test_op_shape_format_fp16(self, device="npu"):
        format_list = [-1]
        shape_list = [1, (32, 3, 3)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        self.dropout_list_exec(shape_format)

    def test_op_shape_format_fp32(self, device="npu"):
        format_list = [-1]
        shape_list = [1, (32, 3, 3)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        self.dropout_list_exec(shape_format)

    def test_will_engine_execute_node(self):

        def get_grad_fn(t):
            if t.requires_grad and t.grad_fn is None:
                return t.clone().grad_fn.next_functions[0][0]
            else:
                return t.grad_fn

        m = torch.nn.Dropout(0.1)
        a = torch.randn(2, 3, 4, requires_grad=True).npu()
        b = m(a)
        should_execute = list(map(get_grad_fn, (a, b)))

        def fn(x):
            for g in should_execute:
                self.assertTrue(torch._C._will_engine_execute_node(g))

        a.register_hook(fn)
        out = b.sum()
        torch.autograd.backward(out, inputs=(a), retain_graph=True)


if __name__ == "__main__":
    run_tests()
