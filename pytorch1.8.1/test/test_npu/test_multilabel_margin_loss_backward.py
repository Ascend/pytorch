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
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from itertools import repeat, product

class TestMultilabelMarginLossGrad(TestCase):

    def generate_data(self, lo, hi, shape, dtype):
        grad = np.random.uniform(lo, hi, (shape[0],)).astype(dtype)
        predict = np.random.uniform(lo, hi, shape).astype(dtype)
        npu_grad = torch.from_numpy(grad)
        npu_predict = torch.from_numpy(predict)
        return npu_grad, npu_predict
    
    def generate_target(self, lo, hi, shape, dtype):
        target = np.random.randint(lo, hi, shape).astype(dtype)
        npu_target = torch.from_numpy(target)
        return npu_target
    
    def cpu_op_grad_exec(self, grad_output, predict, target, reduction):
        predict.requires_grad = True
        target = target.to(torch.int64)
        out = torch.nn.functional.multilabel_margin_loss(input=predict, target=target, reduction=reduction)
        if reduction == "none":
            out.backward(grad_output)
        else:
            out.backward()
        output = predict.grad.to(torch.float32).numpy()
        return output
 
    def npu_op_grad_exec(self, grad_output, predict, target, reduction):
        grad_output = grad_output.to("npu")
        predict = predict.to("npu")
        target = target.to("npu")
        predict.requires_grad = True
        out = torch.nn.functional.multilabel_margin_loss(input=predict, target=target, reduction=reduction)
        if reduction == "none":
            out.backward(grad_output)
        else:
            out.backward()
        output = predict.grad.to("cpu").to(torch.float32).numpy()
        return output

    def test_multilabel_margin_loss_1(self, device):
        for reduction in ["none", "mean", "sum"]:
            grad, data = self.generate_data(-2, 2, (2, 4), np.float32)
            target = self.generate_target(-1, 3, (2, 4), np.int32)
            
            data.requires_grad = False
            cpu_output = self.cpu_op_grad_exec(grad, data, target, reduction)
            data.requires_grad = False
            npu_output = self.npu_op_grad_exec(grad, data, target, reduction)
            
            self.assertRtolEqual(cpu_output, npu_output)

    def test_multilabel_margin_loss_2(self, device):
        for reduction in ["mean", "none", "sum"]:
            grad, data = self.generate_data(-2, 2, (2, 9), np.float32)
            target = self.generate_target(-1, 8, (2, 9), np.int32)
            
            data.requires_grad = False
            cpu_output = self.cpu_op_grad_exec(grad, data, target, reduction)
            data.requires_grad = False
            npu_output = self.npu_op_grad_exec(grad, data, target, reduction)
            
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_multilabel_margin_loss_3(self, device):
        for reduction in ["mean", "none", "sum"]:
            grad, data = self.generate_data(-2, 2, (64, 147), np.float32)
            target = self.generate_target(-1, 146, (64, 147), np.int32)
            
            data.requires_grad = False
            cpu_output = self.cpu_op_grad_exec(grad, data, target, reduction)
            data.requires_grad = False
            npu_output = self.npu_op_grad_exec(grad, data, target, reduction)
            
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_multilabel_margin_loss_float16_1(self, device):
        for reduction in ["mean", "none", "sum"]:
            grad, data = self.generate_data(-2, 2, (2, 4), np.float16)
            target = self.generate_target(-1, 3, (2, 4), np.int32)

            data.requires_grad = False
            grad = grad.to(torch.float32)
            data = data.to(torch.float32)
            cpu_output = self.cpu_op_grad_exec(grad, data, target, reduction)
            data.requires_grad = False
            grad = grad.to(torch.float16)
            data = data.to(torch.float16)
            npu_output = self.npu_op_grad_exec(grad, data, target, reduction)
            
            cpu_output = cpu_output.astype(np.float16)
            npu_output = npu_output.astype(np.float16)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_multilabel_margin_loss_float16_2(self, device):
        for reduction in ["mean", "none", "sum"]:
            grad, data = self.generate_data(-2, 2, (2, 9), np.float16)
            target = self.generate_target(-1, 8, (2, 9), np.int32)
            
            data.requires_grad = False
            grad = grad.to(torch.float32)
            data = data.to(torch.float32)
            cpu_output = self.cpu_op_grad_exec(grad, data, target, reduction)
            data.requires_grad = False
            grad = grad.to(torch.float16)
            data = data.to(torch.float16)
            npu_output = self.npu_op_grad_exec(grad, data, target, reduction)
            
            cpu_output = cpu_output.astype(np.float16)
            npu_output = npu_output.astype(np.float16)

            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_multilabel_margin_loss_float16_3(self, device):
        for reduction in ["mean", "none", "sum"]:
            grad, data = self.generate_data(-2, 2, (1, 79), np.float16)
            target = self.generate_target(-1, 50, (1, 79), np.int32)
            
            data.requires_grad = False
            grad = grad.to(torch.float32)
            data = data.to(torch.float32)
            cpu_output = self.cpu_op_grad_exec(grad, data, target, reduction)
            data.requires_grad = False
            grad = grad.to(torch.float16)
            data = data.to(torch.float16)
            npu_output = self.npu_op_grad_exec(grad, data, target, reduction)
            
            cpu_output = cpu_output.astype(np.float16)
            npu_output = npu_output.astype(np.float16)

            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_multilabel_margin_loss_float16_4(self, device):
        for reduction in ["none", "sum", "mean"]:
            grad, data = self.generate_data(-2, 2, (64, 147), np.float16)
            target = self.generate_target(-1, 146, (64, 147), np.int32)
            
            data.requires_grad = False
            grad = grad.to(torch.float32)
            data = data.to(torch.float32)
            cpu_output = self.cpu_op_grad_exec(grad, data, target, reduction)
            data.requires_grad = False
            grad = grad.to(torch.float16)
            data = data.to(torch.float16)
            npu_output = self.npu_op_grad_exec(grad, data, target, reduction)
            
            cpu_output = cpu_output.astype(np.float16)
            npu_output = npu_output.astype(np.float16)

            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestMultilabelMarginLossGrad, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
