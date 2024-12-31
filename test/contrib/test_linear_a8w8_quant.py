import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.module import LinearA8W8Quant

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestLinearA8W8Quant(TestCase):

    def npu_linear_quant(self, in_features, out_features, x1, x2, scale):
        model = LinearA8W8Quant(in_features, out_features, bias=False, pertoken_scale=False, offset=False)
        model = model.npu()
        model.weight.data = x2
        model.scale.data = scale
        output = model(x1)
        return output

    @unittest.skipIf(DEVICE_NAME == 'Ascend910A' or DEVICE_NAME == 'Ascend310P',
        "OP `QuantBatchMatmulV3` is not supported on 910A or 310P, skip this ut for this device type!")
    def test_npu_linear_quant(self):
        x1 = torch.randint(-1, 1, (1, 5), dtype=torch.int8).npu()
        x2 = torch.randint(-1, 1, (127, 5), dtype=torch.int8).npu()
        scale = torch.randn(1, dtype=torch.float32).npu()
        supported_output = torch_npu.npu_quant_matmul(x1, x2.t(), scale)
        in_features = 5
        out_features = 127
        npu_out = self.npu_linear_quant(in_features, out_features, x1, x2, scale)
        self.assertRtolEqual(supported_output, npu_out, 0.001)

    @unittest.skipIf(DEVICE_NAME == 'Ascend910A' or DEVICE_NAME == 'Ascend310P',
        "OP `QuantBatchMatmulV3` is not supported on 910A or 310P, skip this ut for this device type!")
    def test_npu_linear_quant_out_bf16(self):
        x1 = torch.randint(-1, 1, (1, 5), dtype=torch.int8).npu()
        x2 = torch.randint(-1, 1, (127, 5), dtype=torch.int8).npu()
        scale = torch.randn(1, dtype=torch.float32).npu()
        out_dtype = torch.bfloat16
        supported_output = torch_npu.npu_quant_matmul(x1, x2.t(), scale, output_dtype=out_dtype)
        in_features = 5
        out_features = 127
        model = LinearA8W8Quant(in_features, out_features, bias=False, offset=False, pertoken_scale=False, output_dtype=out_dtype)
        model.weight.data = x2
        model.scale.data = scale
        npu_out = model(x1)
        self.assertRtolEqual(supported_output, npu_out, 0.001)

    @unittest.skipIf(DEVICE_NAME == 'Ascend910A' or DEVICE_NAME == 'Ascend310P',
        "OP `QuantBatchMatmulV3` is not supported on 910A or 310P, skip this ut for this device type!")
    @unittest.expectedFailure
    def test_npu_linear_quant_out_int32_error(self):
        x1 = torch.randint(-1, 1, (1, 5), dtype=torch.int8).npu()
        x2 = torch.randint(-1, 1, (127, 5), dtype=torch.int8).npu()
        scale = torch.randn(1, dtype=torch.float32).npu()
        out_dtype = torch.int32
        supported_output = torch_npu.npu_quant_matmul(x1, x2.t(), scale, output_dtype=out_dtype)
        in_features = 5
        out_features = 127
        model = LinearA8W8Quant(in_features, out_features, bias=False, offset=False, pertoken_scale=False, output_dtype=out_dtype)
        model.weight.data = x2
        model.scale.data = scale
        npu_out = model(x1)
        self.assertRtolEqual(supported_output, npu_out, 0.001)

    @unittest.skipIf(DEVICE_NAME == 'Ascend910A' or DEVICE_NAME == 'Ascend310P',
        "OP `QuantBatchMatmulV3` is not supported on 910A or 310P, skip this ut for this device type!")
    @unittest.expectedFailure
    def test_npu_linear_quant_scale_fp16_error(self):
        x1 = torch.randint(-1, 1, (1, 5), dtype=torch.int8).npu()
        x2 = torch.randint(-1, 1, (127, 5), dtype=torch.int8).npu()
        scale = torch.randn(1, dtype=torch.float16).npu()
        out_dtype = torch.int8
        supported_output = torch_npu.npu_quant_matmul(x1, x2.t(), scale, output_dtype=out_dtype)
        in_features = 5
        out_features = 127
        model = LinearA8W8Quant(in_features, out_features, bias=False, offset=False, pertoken_scale=False, output_dtype=out_dtype)
        model.weight.data = x2
        model.scale.data = scale
        npu_out = model(x1)
        self.assertRtolEqual(supported_output, npu_out, 0.001)

if __name__ == "__main__":
    run_tests()