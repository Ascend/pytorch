import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.module import LinearA8W8Quant
from torch_npu.testing.common_utils import SupportedDevices

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestLinearA8W8Quant(TestCase):

    def npu_linear_quant(self, in_features, out_features, x1, x2, scale):
        model = LinearA8W8Quant(in_features, out_features, False)
        model = model.npu()
        model.weight.data = x2
        model.scale.data = scale
        output = model(x1)
        return output

    @SupportedDevices(['Ascend910B'])
    def test_npu_linear_quant(self):
        x1 = torch.randint(-1, 1, (1, 5), dtype=torch.int8).npu()
        x2 = torch.randint(-1, 1, (5, 127), dtype=torch.int8).npu()
        scale = torch.randn(1, dtype=torch.float32).npu()
        supported_output = torch_npu.npu_quant_matmul(x1, x2, scale)
        in_features = 5
        out_features = 127
        npu_out = self.npu_linear_quant(in_features, out_features, x1, x2, scale)
        self.assertRtolEqual(supported_output, npu_out, 0.001)

if __name__ == "__main__":
    run_tests()