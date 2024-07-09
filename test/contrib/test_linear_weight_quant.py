import unittest
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.module import LinearWeightQuant

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def f32_2_s9(array):
    array_round = np.round(array)
    array_round_clip = np.clip(array_round, -256, 255)
    return array_round_clip


class TestLinearWeightQuant(TestCase):
    def cpu_linear_weight_quant(self, weight_cpu, x_cpu, antiquant_scale_cpu, antiquant_offset_cpu):
        x_cpu = x_cpu.to(torch.float32)
        antiquant_weight = (weight_cpu + antiquant_offset_cpu) * antiquant_scale_cpu
        antiquant_weight = antiquant_weight.to(torch.float32)
        cpu_out = torch.matmul(x_cpu, antiquant_weight).numpy()
        cpu_out = cpu_out.astype("float16")
        return cpu_out

    def npu_linear_weight_quant(self, in_features, out_features, antiquant_scale, weight, x, antiquant_offset=None):
        model = LinearWeightQuant(in_features,
                                  out_features,
                                  bias=False,
                                  device=torch.device(f'npu:0'),
                                  dtype=x.dtype,
                                  antiquant_offset=True,
                                  quant_scale=False,
                                  quant_offset=False,
                                  antiquant_group_size=0
                                  )
        model = model.npu()
        model.weight.data = weight
        model.antiquant_scale.data = antiquant_scale
        model.antiquant_offset.data = antiquant_offset
        npu_out = model(x)

        return npu_out

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
        "OP `WeightQuantBatchMatmulV2` is only supported on 910B, skip this ut for this device type!")
    def test_npu_linear_weight_quant(self):
        m = 1024
        k = 11264
        n = 1664
        x_cpu = torch.randn((m, k), dtype=torch.float16)
        weight_cpu = torch.randn((n, k), dtype=torch.float16)
        weight_cpu = weight_cpu.to(torch.int8)
        weight_cpu = weight_cpu.to(torch.float16)
        weight_cpu_trans = weight_cpu.transpose(0, 1)
        antiquant_scale_cpu = torch.randn((n), dtype=torch.float16)
        antiquant_offset_cpu = torch.randn((n), dtype=torch.float16)

        x_npu = x_cpu.npu()
        weight_npu = weight_cpu.to(torch.int8).npu()
        antiquant_scale_npu = antiquant_scale_cpu.npu()
        antiquant_offset_npu = antiquant_offset_cpu.npu()

        npu_out = self.npu_linear_weight_quant(k, n, antiquant_scale_npu, weight_npu, x_npu, antiquant_offset_npu)
        cpu_out = self.cpu_linear_weight_quant(weight_cpu_trans, x_cpu, antiquant_scale_cpu, antiquant_offset_cpu)
        npu_out = npu_out.cpu()
        self.assertRtolEqual(cpu_out, npu_out.numpy(), 0.01)

if __name__ == "__main__":
    run_tests()