import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestMinNamesDim(TestCase):
    def cpu_op_exec(self, data, dim, keepdim=False):
        outputs, indices = torch.min(data, dim, keepdim)
        return outputs.cpu().numpy(), indices.int().cpu().numpy()

    def npu_op_exec(self, data, dim, keepdim=False):
        outputs, indices = torch_npu.npu_min(data, dim, keepdim)
        return outputs.cpu().numpy(), indices.cpu().numpy()

    def test_min_names_dim_without_keepdim(self):
        data = torch.randn(2, 3, 4, 5, dtype=torch.float32,
                           names=('A', 'B', 'C', 'D')).npu()
        cpu_value, cpu_indices = self.cpu_op_exec(data, 1)
        npu_value, npu_indices = self.npu_op_exec(data, 'B')
        self.assertRtolEqual(cpu_value, npu_value)
        self.assertRtolEqual(cpu_indices, npu_indices)

    def test_min_names_dim_with_keepdim(self):
        data = torch.randn(2, 3, 4, 5, dtype=torch.float32,
                           names=('A', 'B', 'C', 'D')).npu()
        cpu_value, cpu_indices = self.cpu_op_exec(data, 3, True)
        npu_value, npu_indices = self.npu_op_exec(data, 'D', True)
        self.assertRtolEqual(cpu_value, npu_value)
        self.assertRtolEqual(cpu_indices, npu_indices)


if __name__ == "__main__":
    run_tests()
