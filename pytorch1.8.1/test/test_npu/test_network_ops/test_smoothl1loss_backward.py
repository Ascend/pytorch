# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class Testcdist(TestCase):
    def generate_data(self, min_n, max_n, shape_predict, shape_label, src_type):
        np.random.seed(10086)
        predict = np.random.uniform(min_n, max_n, shape_predict).astype(src_type)
        label = np.random.uniform(min_n, max_n, shape_label).astype(src_type)
        label[label < 0] = -1
        label[label >= 0] = 1
        return predict, label

    def op_exec(self, predict, label, reduction, device='cpu'):
        is_fp16 = predict.dtype == np.float16
        if device == 'cpu' and is_fp16:
            predict = predict.astype(np.float32)
            label = label.astype(np.float32)
        predict = torch.from_numpy(predict)
        label = torch.from_numpy(label)        
        predict = predict.to(device)
        label = label.to(device)

        predict.requires_grad = True
        output_forward = F.smooth_l1_loss(predict, label, reduction=reduction)
        output_forward = output_forward.sum()
        output_forward.backward()

        gradient = predict.grad.cpu().numpy()       
        if device == 'cpu' and is_fp16:
            gradient = gradient.astype(np.float16)
        return gradient

    def test_smooth_l1_loss_backward_float16_3(self, device):
        shape_format = [
            [-1, 1, [100], [100], np.float16],
            [-0.1, 0.1, [100, 200], [100, 200], np.float16],
            [-10, 10, [100, 20, 30], [100, 20, 1], np.float16],
            [-0.01, 0.01, [100, 20, 30], [100, 20, 30], np.float16],
            [-0.001, 0.001, [10,20,30,4], [10,20,30,4], np.float16],
            [-0.001, 0.001, [10,20,3,4,5], [10,20,3,4,5], np.float16],
        ]
        for item in shape_format:
            input1, input2 = self.generate_data(item[0], item[1], item[2], item[3], item[4])
            cpu_output1 = self.op_exec(input1, input2, 'none','cpu')
            npu_output1 = self.op_exec(input1, input2, 'none','npu')
            cpu_output2 = self.op_exec(input1, input2, 'mean','cpu')
            npu_output2 = self.op_exec(input1, input2, 'mean','npu')
            cpu_output3 = self.op_exec(input1, input2, 'sum','cpu')
            npu_output3 = self.op_exec(input1, input2, 'sum','npu')
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)
            self.assertRtolEqual(cpu_output3, npu_output3)

    def test_smooth_l1_loss_backward_float32_3(self, device):
        shape_format = [
            [-1, 1, [100], [100], np.float32],
            [-0.1, 0.1, [100, 200], [100, 200], np.float32],
            [-10, 10, [100, 20, 30], [100, 20, 1], np.float32],
            [-0.01, 0.01, [100, 20, 30], [100, 20, 30], np.float32],
            [-0.001, 0.001, [10,20,30,4], [10,20,30,4], np.float32],
            [-0.001, 0.001, [10,20,3,4,5], [10,20,3,4,5], np.float32],
        ]
        for item in shape_format:
            input1, input2 = self.generate_data(item[0], item[1], item[2], item[3], item[4])
            cpu_output1 = self.op_exec(input1, input2, 'none','cpu')
            npu_output1 = self.op_exec(input1, input2, 'none','npu')
            cpu_output2 = self.op_exec(input1, input2, 'mean','cpu')
            npu_output2 = self.op_exec(input1, input2, 'mean','npu')
            cpu_output3 = self.op_exec(input1, input2, 'sum','cpu')
            npu_output3 = self.op_exec(input1, input2, 'sum','npu')
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)
            self.assertRtolEqual(cpu_output3, npu_output3)

instantiate_device_type_tests(Testcdist, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
