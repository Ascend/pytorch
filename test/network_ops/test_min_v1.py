import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestMinV1(TestCase):
    def cpu_op_exec(self, data, dim):
        outputs, indices = torch.min(data, dim)
        return outputs.detach()

    def npu_op_exec(self, data, dim):
        data = data.to("npu")
        outputs, indices = torch_npu.npu_min(data, dim)
        return outputs.detach().cpu()

    def test_min_v1_fp32(self, device="npu"):
        data = torch.randn(2, 2, 2, 2, dtype=torch.float32)
        npu_data = data.clone()
        cpu_out = self.cpu_op_exec(data, 2)
        npu_out = self.npu_op_exec(npu_data, 2)
        self.assertRtolEqual(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()
