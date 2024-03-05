import random
import unittest
import torch
import numpy as np
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestScaledMaskedSoftmax(TestCase):
    def cpu_op_exec_forward(self, x, mask, scale, fixed_triu_mask):
        x = x.float()
        if fixed_triu_mask:
            mask_tri = torch.triu(torch.ones(mask.shape, device=mask.device), diagonal=1).bool()
            output = F.softmax((x * scale).masked_fill(mask_tri, value=-10000), dim=-1).half()
        else:
            output = F.softmax((x * scale).masked_fill(mask, value=-10000), dim=-1).half()
        return output.detach().numpy()

    def npu_op_exec_forward(self, x, mask, scale, fixed_triu_mask):
        output = torch_npu.npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask)
        return output.cpu().detach().numpy()

    def cpu_op_exec_backward(self, x, y_grad, mask, scale, fixed_triu_mask):
        x.requires_grad_(True)
        x_fp32 = x.float()
        y_grad = y_grad.float()
        if fixed_triu_mask:
            mask_tri = torch.triu(torch.ones(mask.shape, device=mask.device), diagonal=1).bool()
            output = F.softmax((x_fp32 * scale).masked_fill(mask_tri, value=-10000), dim=-1).half()
            output.backward(y_grad)
        else:
            output = F.softmax((x_fp32 * scale).masked_fill(mask, value=-10000), dim=-1).half()
            output.backward(y_grad)
        x_grad = x.grad
        return x_grad.detach().numpy()

    def npu_op_exec_backward(self, x, y_grad, mask, scale, fixed_triu_mask):
        x.requires_grad_(True)
        output = torch_npu.npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask)
        output.backward(y_grad)
        x_grad = x.grad
        return x_grad.half().cpu().detach().numpy()

    @unittest.skip("skip test_scaled_masked_softmax_shape_format now")
    def test_scaled_masked_softmax_shape_format(self):
        shape_format = [
            [[np.float16, 29, (16, 6, 128, 128)], [np.float16, 29, (16, 6, 128, 128)]],
            [[np.float16, 2, (16, 6, 128, 512)], [np.float16, 2, (16, 1, 128, 512)]],
            [[np.float16, 0, (16, 6, 512, 512)], [np.float16, 0, (16, 1, 512, 512)]],
            [[np.float32, 29, (16, 6, 128, 128)], [np.float32, 29, (16, 6, 128, 128)]],
            [[np.float32, 2, (16, 6, 128, 512)], [np.float32, 2, (16, 1, 128, 512)]],
            [[np.float32, 0, (16, 6, 512, 512)], [np.float32, 0, (16, 1, 512, 512)]],
        ]

        # forward ut test
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -5, 5)
            cpu_mask, npu_mask = create_common_tensor(item[1], -1, 1)
            cpu_mask = cpu_mask > 0
            npu_mask = npu_mask > 0
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            scale = random.uniform(-1, 1)
            fixed_triu_mask = False
            cpu_output = self.cpu_op_exec_forward(cpu_input, cpu_mask,
                                                  scale, fixed_triu_mask)
            npu_output = self.npu_op_exec_forward(npu_input, npu_mask,
                                                  scale, fixed_triu_mask)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

        # backward ut test
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -5, 5)
            cpu_y_grad, npu_y_grad = create_common_tensor(item[0], -5, 5)
            cpu_mask, npu_mask = create_common_tensor(item[1], -1, 1)
            cpu_mask = cpu_mask > 0
            npu_mask = npu_mask > 0
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            if cpu_y_grad.dtype == torch.float16:
                cpu_y_grad = cpu_y_grad.to(torch.float32)
            scale = random.uniform(-1, 1)
            fixed_triu_mask = False
            cpu_x_grad = self.cpu_op_exec_backward(cpu_input, cpu_y_grad,
                                                   cpu_mask, scale, fixed_triu_mask)
            npu_x_grad = self.npu_op_exec_backward(npu_input, npu_y_grad,
                                                   npu_mask, scale, fixed_triu_mask)
            cpu_x_grad = cpu_x_grad.astype(npu_x_grad.dtype)
            self.assertRtolEqual(cpu_x_grad, npu_x_grad)


if __name__ == "__main__":
    run_tests()
