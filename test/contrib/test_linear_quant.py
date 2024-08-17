import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.module import LinearQuant

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestLinearQuant(TestCase):

    def npu_linear_quant(self, in_features, out_features, x1, x2, scale):
        model = LinearQuant(in_features, out_features, bias=False, pertoken_scale=False, offset=False)
        model = model.npu()
        model.weight.data = x2
        model.scale.data = scale
        output = model(x1)
        return output

    @unittest.skipIf(DEVICE_NAME == 'Ascend910A' or DEVICE_NAME == 'Ascend310P',
        "OP `QuantBatchMatmulV3` is not supported on 910A and 310P, skip this ut for this device type!")
    def test_npu_linear_quant(self):
        x1 = torch.randint(-1, 1, (1, 2), dtype=torch.int32).npu()
        x2 = torch.randint(-1, 1, (128, 2), dtype=torch.int32).npu()
        scale = torch.randn(1, dtype=torch.float32).npu()
        supported_output = torch_npu.npu_quant_matmul(x1, x2.t(), scale)
        in_features = 2
        out_features = 128
        npu_out = self.npu_linear_quant(in_features, out_features, x1, x2, scale)
        self.assertRtolEqual(supported_output, npu_out, 0.001)

if __name__ == "__main__":
    run_tests()