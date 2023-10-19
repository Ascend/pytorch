import torch
import torch.nn.functional as F
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestEmbeddingBackward(TestCase):
    def cpu_op_exec(self, weight, indices):
        weight.requires_grad_(True)
        out = F.embedding(indices, weight, scale_grad_by_freq=False, padding_idx=-1)
        out.backward(torch.ones_like(out))
        grad_cpu = weight.grad
        return out.detach().numpy(), grad_cpu.detach().numpy()

    def npu_op_exec(self, weight, indices):
        weight.requires_grad_(True)
        out = F.embedding(indices, weight, scale_grad_by_freq=False, padding_idx=-1)
        out.backward(torch.ones_like(out))
        out_npu = out.to("cpu")
        grad_npu = weight.grad
        grad_npu = grad_npu.to("cpu")
        return out_npu.detach().numpy(), grad_npu.detach().numpy()

    def test_embedding_backward_shape_format_fp32(self, device="npu"):
        format_list = [0]
        shape_list1 = [[40, 32], [40, 1024], [40000, 1024], [33712, 1024]]
        shape_list2 = [[40], [40], [40000], [33712]]
        shape_format1 = [
            [np.float32, i, j] for i in format_list for j in shape_list1
        ]
        shape_format2 = [
            [np.int64, i, j] for i in format_list for j in shape_list2
        ]
        shape_format = [
            [shape_format1[i], shape_format2[i]] for i in range(len(shape_list1))
        ]
        for item in shape_format:
            weight_cpu, weight_npu = create_common_tensor(item[0], 1, 1)
            indices_cpu, indices_npu = create_common_tensor(item[1], 0, min(item[0][2][0:-1]))

            cpu_out, cpu_grad = self.cpu_op_exec(weight_cpu, indices_cpu)
            npu_out, npu_grad = self.npu_op_exec(weight_npu, indices_npu)

            self.assertRtolEqual(cpu_out, npu_out)
            self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
