import numpy as np
import torch

import torch_npu
from torch_npu.testing._testcase import TestCase, run_tests
from torch_npu.testing._internal.common_utils import create_common_tensor


class TestDtypeCast(TestCase):

    def supported_op_exec(self, input1, dst_dtype):
        output = input1.to(dst_dtype)
        return output.cpu().detach()

    def custom_op_exec(self, input1, dst_dtype):
        output = torch_npu.npu_dtype_cast(input1, dst_dtype)
        return output.cpu().detach()

    def test_npu_dtype_cast(self, device="npu"):
        item = [np.float32, 0, (64, 10)]
        _, npu_input = create_common_tensor(item, -1, 1)
        dst_dtype = torch.float16

        supported_output = self.supported_op_exec(npu_input, dst_dtype)
        custom_output = self.custom_op_exec(npu_input, dst_dtype)
        self.assertRtolEqual(supported_output, custom_output)

    def test_npu_dtype_cast_double_backward(self, device="npu"):
        x = torch.randn(3, 3, requires_grad=True).to(device)
        y = torch_npu.npu_dtype_cast(x, torch.half)
        z = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y))
        self.assertIsNone(z[0].grad_fn)
        z = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True)
        self.assertIsNotNone(z[0].grad_fn)


if __name__ == "__main__":
    run_tests()
