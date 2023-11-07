import torch
import numpy as np
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestEmbedding(TestCase):
    def cpu_op_exec(self, weight, indices):
        weight.requires_grad_(True)
        out = F.embedding(indices, weight, scale_grad_by_freq=True, padding_idx=37)
        return out.detach().numpy()

    def npu_op_exec(self, weight, indices):
        weight.requires_grad_(True)
        out = F.embedding(indices, weight, scale_grad_by_freq=True, padding_idx=37)
        out_npu = out.to("cpu")
        return out_npu.detach().numpy()

    def test_shape_nz_format(self):
        shape_format = [
            [[np.float32, 29, [40, 32]], [np.int64, 0, [40]]],
            [[np.float32, 29, [40, 1024]], [np.int64, 0, [40]]],
            [[np.float32, 29, [40000, 1024]], [np.int64, 0, [3125]]],
            [[np.float32, 29, [40000, 1024]], [np.int64, 0, [128, 8]]],
            [[np.float16, 29, [40, 32]], [np.int64, 0, [40]]],
            [[np.float16, 29, [40, 1024]], [np.int64, 0, [128, 8]]],
            [[np.float16, 29, [33712, 1024]], [np.int64, 0, [64, 7]]],
            [[np.float32, 29, [40, 32]], [np.int64, 0, [40]]],
            [[np.float32, 29, [40, 1024]], [np.int64, 0, [40]]],
            [[np.float32, 29, [40000, 1024]], [np.int64, 0, [3125]]],
            [[np.float32, 29, [40000, 1024]], [np.int64, 0, [128, 8]]],
            [[np.float16, 29, [40, 32]], [np.int64, 0, [40]]],
            [[np.float16, 29, [40, 1024]], [np.int64, 0, [128, 8]]],
            [[np.float16, 29, [33712, 1024]], [np.int64, 0, [64, 7]]]
        ]
        for item in shape_format:
            weight_cpu, weight_npu = create_common_tensor(item[0], 1, 1)
            indices_cpu, indices_npu = create_common_tensor(item[1], 0, 1)

            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)

            cpu_out = self.cpu_op_exec(weight_cpu, indices_cpu)
            npu_out = self.npu_op_exec(weight_npu, indices_npu)
            cpu_out = cpu_out.astype(npu_out.dtype)

            self.assertEqual(cpu_out, npu_out)

    def test_shape_format(self):
        shape_format = [
            [[np.float32, 0, [40, 32]], [np.int64, 0, [40]]],
            [[np.float32, 0, [40, 1024]], [np.int64, 0, [40]]],
            [[np.float32, 0, [40000, 1024]], [np.int64, 0, [3125]]],
            [[np.float32, 0, [40000, 1024]], [np.int64, 0, [128, 8]]],
            [[np.float16, 0, [40, 32]], [np.int64, 0, [40]]],
            [[np.float16, 0, [40, 1024]], [np.int64, 0, [128, 8]]],
            [[np.float16, 0, [33712, 1024]], [np.int64, 0, [64, 7]]],
            [[np.float32, -1, [40, 32]], [np.int64, 0, [40]]],
            [[np.float32, -1, [40, 1024]], [np.int64, 0, [40]]],
            [[np.float32, -1, [40000, 1024]], [np.int64, 0, [3125]]],
            [[np.float32, -1, [40000, 1024]], [np.int64, 0, [128, 8]]],
            [[np.float16, -1, [40, 32]], [np.int64, 0, [40]]],
            [[np.float16, -1, [40, 1024]], [np.int64, 0, [128, 8]]],
            [[np.float16, -1, [33712, 1024]], [np.int64, 0, [64, 7]]]
        ]
        for item in shape_format:
            weight_cpu, weight_npu = create_common_tensor(item[0], 1, 1)
            indices_cpu, indices_npu = create_common_tensor(item[1], 0, 1)

            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)

            cpu_out = self.cpu_op_exec(weight_cpu, indices_cpu)
            npu_out = self.npu_op_exec(weight_npu, indices_npu)
            cpu_out = cpu_out.astype(npu_out.dtype)

            self.assertEqual(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()
