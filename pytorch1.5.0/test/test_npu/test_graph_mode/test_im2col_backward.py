# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import numpy as np
import sys
import copy
import os
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from graph_utils import graph_mode

class TestIm2colBackward(TestCase):
    @graph_mode
    def test_im2col_backward_fp16(self, device):
        fold_cpu = torch.nn.Fold(output_size=(18, 18), kernel_size=(3, 3))
        input_cpu = torch.rand(1, 16 * 3 * 3, 256).half()
        fold_npu = fold_cpu.npu()
        input_npu = input_cpu.npu()
        output_cpu = fold_cpu(input_cpu)
        output_npu = fold_npu(input_npu)

        self.assertRtolEqual(output_cpu.numpy(), output_npu.cpu().numpy())

    @graph_mode
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