import torch
import numpy as np
import sys
import copy
import os
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestIm2colBackward(TestCase):
    def test_im2col_backward_fp16(self, device):
        fold_cpu = torch.nn.Fold(output_size=(18, 18), kernel_size=(3, 3))
        input_cpu = torch.rand(1, 16 * 3 * 3, 256).half()
        fold_npu = fold_cpu.npu()
        input_npu = input_cpu.npu()
        output_cpu = fold_cpu(input_cpu)
        output_npu = fold_npu(input_npu)

        self.assertRtolEqual(output_cpu.numpy(), output_npu.cpu().numpy())

    def test_im2col_backward_fp32(self, device):
        fold_cpu = torch.nn.Fold(output_size=(18, 18), kernel_size=(3, 3))
        input_cpu = torch.rand(1, 16 * 3 * 3, 256)
        fold_npu = fold_cpu.npu()
        input_npu = input_cpu.npu()
        output_cpu = fold_cpu(input_cpu).numpy()
        output_npu = fold_npu(input_npu).cpu().numpy()

        self.assertRtolEqual(output_cpu, output_npu)


instantiate_device_type_tests(TestIm2colBackward, globals(), except_for='cpu')
if __name__ == '__main__':
    run_tests()