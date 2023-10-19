import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class Testcdist(TestCase):

    def generate_data(self, min_n, max_n, shape_x, shape_y, src_type):
        np.random.seed(10086)
        x1 = np.random.uniform(min_n, max_n, shape_x).astype(src_type)
        x2 = np.random.uniform(min_n, max_n, shape_y).astype(src_type)
        return x1, x2

    def cdist_backward(self, x1, x2, p, grad, cdist):
        x1 = torch.unsqueeze(x1, -2)
        x2 = torch.unsqueeze(x2, -3)
        grad = torch.unsqueeze(grad, -1)
        cdist = torch.unsqueeze(cdist, -1)
        diff = x1 - x2
        diff_abs = torch.abs(diff)
        nz_cdist = torch.where(cdist == 0, torch.ones_like(cdist), cdist)
        sign = torch.where(diff > 0, torch.ones_like(diff), torch.full_like(diff, -1))
        sign = torch.where(diff == 0, torch.zeros_like(diff), sign)

        if p == 0.0:
            res = torch.zeros_like(diff)
        elif p == 1.0:
            res = grad * sign
        elif p < 2.0:
            res = sign * torch.pow(diff_abs, p - 1.0) * grad / torch.pow(nz_cdist, p - 1.0)
            res = torch.where(cdist == 0, torch.zeros_like(res), res)
        elif p == 2.0:
            res = grad * diff / nz_cdist
            res = torch.where(cdist == 0, torch.zeros_like(res), res)
        elif p == float("inf"):
            mask = torch.where(cdist - diff_abs > 0, torch.zeros_like(diff), torch.ones_like(diff))
            res = grad * sign * mask
        else:
            res = diff * torch.pow(diff_abs, p - 2) * grad / torch.pow(nz_cdist, p - 1.0)
        res = torch.where(cdist == 0, torch.zeros_like(res), res)
        res = torch.sum(res, -2)
        return res

    def op_exec(self, x1, x2, p, device='cpu'):
        is_fp16 = x1.dtype == np.float16
        if device == 'cpu' and is_fp16:
            x1 = x1.astype(np.float32)
            x2 = x2.astype(np.float32)
        x1 = torch.tensor(x1, device=device, requires_grad=True)
        x2 = torch.tensor(x2, device=device, requires_grad=True)
        y = torch.cdist(x1, x2, p)
        grad = torch.ones_like(y, requires_grad=True, device=device)
        if device == 'cpu' and is_fp16:
            y = y.half()
            y = y.float()
            out = self.cdist_backward(x1, x2, p, grad, y)
            return out.detach().numpy().astype('float16')
        y.backward(grad, retain_graph=True)
        out = x1.grad.detach().cpu().numpy()
        return out

    def test_cdis_backward_common_shape(self):
        shape_items = [
            [np.float16, (5, 10), (4, 10)],
            [np.float16, (20, 5, 10), (20, 4, 10)],
            [np.float32, (5, 10), (4, 10)],
            [np.float32, (20, 5, 10), (20, 4, 10)],
        ]
        p_ranges = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        for item in shape_items:
            for p in p_ranges:
                input1, input2 = self.generate_data(-1, 1,
                                                    item[1], item[2], item[0])
                cpu_output = self.op_exec(input1, input2, p, device='cpu')
                npu_output = self.op_exec(input1, input2, p, device='npu')
                self.assertRtolEqual(cpu_output, npu_output)

    def test_cdis_backward_input_range(self):
        item = [np.float32, (20, 5, 5), (20, 4, 5)]
        p_ranges = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        input_ragnes = [(-0.1, 0.1), (-10, 10), (-20, 20)]
        for p in p_ranges:
            for min_max in input_ragnes:
                input1, input2 = self.generate_data(min_max[0], min_max[1],
                                                    item[1], item[2], item[0])
                cpu_output = self.op_exec(input1, input2, p, device='cpu')
                npu_output = self.op_exec(input1, input2, p, device='npu')
                self.assertRtolEqual(cpu_output, npu_output)

    def test_cdis_backward_inf(self):
        shape_items = [
            [np.float16, (5, 10), (4, 10)],
            [np.float16, (20, 5, 10), (20, 4, 10)],
            [np.float32, (5, 10), (4, 10)],
            [np.float32, (20, 5, 10), (20, 4, 10)],
        ]
        p_ranges = [np.inf]
        for item in shape_items:
            for p in p_ranges:
                input1, input2 = self.generate_data(-1, 1, item[1], item[2], item[0])
                cpu_output = self.op_exec(input1, input2, p, device='cpu')
                npu_output = self.op_exec(input1, input2, p, device='npu')
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
