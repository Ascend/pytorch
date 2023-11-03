import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestMinDim(TestCase):
    def cpu_op_exec(self, data, dim, keepdim=False):
        outputs, indices = torch.min(data, dim, keepdim)
        return outputs.cpu().numpy(), indices.int().cpu().numpy()

    def npu_op_exec(self, data, dim, keepdim=False):
        outputs, indices = torch_npu.npu_min(data, dim, keepdim)
        return outputs.cpu().numpy(), indices.cpu().numpy()

    def test_min_dim_without_keepdim(self):
        data = torch.randn(2, 3, 4, 5, dtype=torch.float32).npu()
        cpu_value, cpu_indices = self.cpu_op_exec(data, 2)
        npu_value, npu_indices = self.npu_op_exec(data, 2)
        self.assertRtolEqual(cpu_value, npu_value)
        self.assertRtolEqual(cpu_indices, npu_indices)

    def test_min_dim_with_keepdim(self):
        data = torch.randn(2, 3, 4, 5, dtype=torch.float32).npu()
        cpu_value, cpu_indices = self.cpu_op_exec(data, 3, True)
        npu_value, npu_indices = self.npu_op_exec(data, 3, True)
        self.assertRtolEqual(cpu_value, npu_value)
        self.assertRtolEqual(cpu_indices, npu_indices)


if __name__ == "__main__":
    run_tests()
