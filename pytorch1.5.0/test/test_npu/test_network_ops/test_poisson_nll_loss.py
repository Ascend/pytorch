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
import sys 
import copy 
import torch 
import numpy as np 
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestPoissonNllLoss(TestCase):

    def generate_data(self, min_num, max_num, shape, dtype): 
        input1 = np.random.uniform(min_num, max_num, shape).astype(dtype) 
        input2 = np.random.uniform(min_num, max_num, shape).astype(dtype) 
     
        #modify from numpy.ndarray to torch.tensor 
        npu_input1 = torch.from_numpy(input1) 
        npu_input2 = torch.from_numpy(input2) 
         
        return npu_input1, npu_input2 
         

    def cpu_op_exec(self, input_x, target, log_input, full, eps, reduction): 
        flag = 0
        if input_x.dtype != torch.float32:
            input_x = input_x.to(torch.float32)
            target = target.to(torch.float32)
            flag = 1

        output = torch.poisson_nll_loss(input_x, target, log_input, full, eps, reduction)
        if flag == 1:
            output = output.to(torch.float16)
        output = output.numpy()
        return output 
     
     
    def npu_op_exec(self, input_x, target, log_input, full, eps, reduction): 
        input_x = input_x.to("npu")
        target = target.to("npu") 
        
        output = torch.poisson_nll_loss(input_x, target, log_input, full, eps, reduction)
        output = output.to("cpu") 
        output = output.numpy()
        return output 
     

    def test_poisson_nll_loss_float16_0_none(self, device):
        eps = 1e-8
        log_input = True
        full = False
        reduction = 0 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (2, 2), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float16_1_mean(self, device):
        eps = 1e-8
        log_input = True
        full = False
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (2, 2), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float16_2_sum(self, device):
        eps = 1e-8
        log_input = True
        full = False
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (2, 2), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float16_3_sum(self, device):
        eps = 1e-8
        log_input = False
        full = False
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (8, 16), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float16_4_mean(self, device):
        eps = 1e-8
        log_input = False
        full = False
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (8, 16), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float16_5_mean(self, device):
        eps = 1e-8
        log_input = False
        full = False
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (8, 16), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float16_6_mean(self, device):
        eps = 1e-8
        log_input = True
        full = True
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 16, (8, 16, 32), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float16_7_none(self, device):
        eps = 1e-8
        log_input = True
        full = True
        reduction = 0 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 32, (8, 16, 32), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float16_8_sum(self, device):
        eps = 1e-8
        log_input = True
        full = True
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 32, (8, 16, 32), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float16_9_none(self, device):
        eps = 1e-8
        log_input = False
        full = True
        reduction = 0 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 64, (2, 4, 8, 16), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float16_10_sum(self, device):
        eps = 1e-8
        log_input = False
        full = True
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 64, (2, 4, 8, 16), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float16_11_mean(self, device):
        eps = 1e-8
        log_input = False
        full = True
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 64, (2, 4, 8, 16), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float16_12_sum(self, device):
        eps = 1.0
        log_input = True
        full = False
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (65500, 1, 1, 1), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float16_13_none(self, device):
        eps = 1.0
        log_input = True
        full = True
        reduction = 0 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (8192, 1, 1, 1), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float16_14_mean(self, device):
        eps = 1.0
        log_input = False
        full = True
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (16384, 1, 1, 1), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float16_15_sum(self, device):
        eps = 1.0
        log_input = False
        full = False
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (32768, 1, 1, 1), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float16_16_none(self, device):
        eps = 1.0
        log_input = False
        full = False
        reduction = 0 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(-2.0, 2.0, (7,1,11,160,1088), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float16_17_mean(self, device):
        eps = 1.0
        log_input = False
        full = False
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(-2.0, 2.0, (7,1,11,160,1088), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float16_18_sum(self, device):
        eps = 1.0
        log_input = False
        full = False
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(-2.0, 2.0, (7,1,11,160,1088), np.float16) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float32_0_none(self, device):
        eps = 1e-8
        log_input = True
        full = False
        reduction = 0 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (1, 31, 149, 2), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float32_1_mean(self, device):
        eps = 1e-8
        log_input = True
        full = False
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (1, 31, 149, 2), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float32_2_sum(self, device):
        eps = 1e-8
        log_input = True
        full = False
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (1, 31, 149, 2), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float32_3_sum(self, device):
        eps = 1e-8
        log_input = False
        full = False
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 3402823500.0, (1, 32, 31, 1), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float32_4_mean(self, device):
        eps = 1e-8
        log_input = False
        full = False
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 0.000030517578125, (2, 32, 149, 31), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float32_5_none(self, device):
        eps = 1e-8
        log_input = False
        full = False
        reduction = 0 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 3402800000, (128), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float32_6_mean(self, device):
        eps = 1e-8
        log_input = True
        full = True
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 9.313225746154785e-10,(128, 1), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float32_7_none(self, device):
        eps = 1e-8
        log_input = True
        full = True
        reduction = 0 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 9.313225746154785e-10, (1, 31, 149, 2), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float32_8_sum(self, device):
        eps = 1e-8
        log_input = True
        full = True
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 16, (1, 1, 1, 16384), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float32_9_none(self, device):
        eps = 1e-8
        log_input = False
        full = True
        reduction = 0 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0,0.000000000000000000000000000000000000011754943508, (2, 31, 149, 2), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float32_10_sum(self, device):
        eps = 1e-8
        log_input = False
        full = True
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0,0.000000000000000000000000000000000000011754943508, (2, 31, 149, 2), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float32_11_mean(self, device):
        eps = 1e-8
        log_input = False
        full = True
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0,0.000000000000000000000000000000000000011754943508, (2, 31, 149, 2), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float32_12_sum(self, device):
        eps = 1.0
        log_input = True
        full = False
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 2, (65535, 1, 1, 1), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float32_13_none(self, device):
        eps = 1.0
        log_input = True
        full = True
        reduction = 0 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 3402823500.0, (1, 32, 31, 1), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float32_14_mean(self, device):
        eps = 1.0
        log_input = False
        full = True
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 3402823500.0, (1, 32, 31, 1), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float32_15_sum(self, device):
        eps = 1.0
        log_input = False
        full = False
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(0, 3402823500.0, (1, 32, 31, 1), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output)

    def test_poisson_nll_loss_float32_16_none(self, device):
        eps = 1.0
        log_input = True
        full = False
        reduction = 0 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(-2.0, 2.0, (7,1,11,160,1088), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float32_17_mean(self, device):
        eps = 1.0
        log_input = True
        full = False
        reduction = 1 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(-2.0, 2.0, (7,1,11,160,1088), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_poisson_nll_loss_float32_18_sum(self, device):
        eps = 1.0
        log_input = True
        full = False
        reduction = 2 # 可选值 0/1/2 分别表示 'none','mean','sum'
        input_x, target = self.generate_data(-2.0, 2.0, (7,1,11,160,1088), np.float32) 
        cpu_output = self.cpu_op_exec(input_x, target, log_input, full, eps, reduction) 
        npu_output = self.npu_op_exec(input_x, target, log_input, full, eps, reduction) 
        self.assertRtolEqual(cpu_output, npu_output) 

instantiate_device_type_tests(TestPoissonNllLoss, globals(), except_for='cpu')    
if __name__ == '__main__': 
    # 当前版本需要调用如下代码 
    torch.npu.set_device("npu:1") 
    run_tests() 

