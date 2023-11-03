import torch
import torch.nn as nn
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLayerNorm(TestCase):
    def test_c10_layer_norm(self, device="npu"):
        # test that we can call c10 ops and they return a reasonable result
        X = torch.rand(5, 5, dtype=torch.float, device="cpu")
        X = X.to("npu")
        weight = torch.rand(*X.size()[1:], dtype=torch.float, device="cpu")
        weight = weight.to("npu")
        bias = torch.rand(*X.size()[1:], dtype=torch.float, device="cpu")
        bias = bias.to("npu")
        epsilon = 1e-4

        expected_norm = torch.nn.functional.layer_norm(
            X, X.size()[1:], weight=weight, bias=bias, eps=epsilon)
        expected_norm_cpu = torch.nn.functional.layer_norm(
            X.cpu(), X.size()[1:], weight=weight.cpu(), bias=bias.cpu(), eps=epsilon)
        self.assertRtolEqual(expected_norm.cpu().numpy(), expected_norm_cpu.numpy())

    def cpu_op_exec(self, input1):
        m = nn.LayerNorm(input1.size()[1:])
        output = m(input1)
        return output

    def npu_op_exec(self, input1):
        m = nn.LayerNorm(input1.size()[1:]).npu()
        output = m(input1)
        output = output.to("cpu")
        return output

    def test_layer_norm_shape_format(self, device="npu"):
        shape_format = [
            [np.float32, 0, (64, 10)],
            [np.float32, 0, (256, 2048, 7, 7)],
            [np.float32, 0, (32, 1, 3, 3)],
            [np.float32, 0, (10, 128)],
            [np.float32, 2, (46, 16)],
            [np.float32, 3, (2, 2, 2)],
            [np.float32, 29, (3, 4, 5, 6)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())

    def test_layer_norm_float16_format(self, device="npu"):
        shape_format = [
            [np.float16, 0, (64, 10)],
            [np.float16, 0, (256, 2048, 7, 7)],
            [np.float16, 0, (32, 1, 3, 3)],
            [np.float16, 0, (10, 128)],
            [np.float16, 2, (46, 16)],
            [np.float16, 3, (2, 2, 2)],
            [np.float16, 29, (3, 4, 5, 6)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.to(torch.float16)
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())

    def test_layer_norm_case_in_trocr(self):
        cpu_input = torch.rand(10, 1, 1024).uniform_(-22., 66.).half()
        cpu_weight = torch.rand(1024).uniform_(0.5, 1.1).half()
        cpu_bias = torch.rand(1024).uniform_(-0.1, 0.1).half()
        npu_input = cpu_input.npu()
        npu_weight = cpu_weight.npu()
        npu_bias = cpu_bias.npu()
        normalized_shape = (1024,)
        eps = 1e-05

        cpu_out1 = torch.layer_norm(cpu_input.float(), normalized_shape, cpu_weight.float(), cpu_bias.float(),
                                    eps, torch.backends.cudnn.enabled).half()
        npu_out1 = torch.layer_norm(npu_input, normalized_shape, npu_weight, npu_bias,
                                    eps, torch.backends.cudnn.enabled)
        self.assertRtolEqual(cpu_out1, npu_out1.cpu())

        cpu_out2 = torch.layer_norm(cpu_input.float(), normalized_shape, cpu_weight.float(), cpu_bias.float(),
                                    eps, torch.backends.cudnn.enabled).half()
        npu_out2 = torch.layer_norm(npu_input, normalized_shape, npu_weight, npu_bias,
                                    eps, torch.backends.cudnn.enabled)
        self.assertRtolEqual(cpu_out2, npu_out2.cpu())


if __name__ == "__main__":
    run_tests()
