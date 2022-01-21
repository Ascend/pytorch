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
import torch_npu
import numpy as np

from torch_npu.testing.common_utils import TestCase, run_tests
from torch_npu.testing.common_device_type import instantiate_device_type_tests
from torch_npu.testing.util_test import create_common_tensor

class TestTopK(TestCase):

    def cpu_op_exec(self, x, k):
        output, indices = torch.topk(x, k)
        output = output.numpy()
        indices = indices.numpy().astype(np.int32)
        return output, indices

    def npu_op_exec_new(self, x, k):
        output, indices = torch.topk(x, k)
        output = output.to("cpu")
        indices = indices.to("cpu")
        output = output.numpy()
        indices = indices.numpy().astype(np.int32)
        return output, indices

    def topk_result(self, shape_format):
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output, cpu_indices = self.cpu_op_exec(cpu_input, 5)
            npu_output, npu_indices = self.npu_op_exec_new(npu_input, 5)
            cpu_output = cpu_output.astype(npu_output.dtype)
            #  目前只支持fp16,fp32降低阈值判断
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-1)
            
    def test_topk_shape_format_fp16_1d(self, device):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [18]] for i in format_list
        ]        
        self.topk_result(shape_format)
        
    def test_topk_shape_format_fp32_1d(self, device):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [18]] for i in format_list
        ]        
        self.topk_result(shape_format)
        
    def test_topk_shape_format_fp16_2d(self, device):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [5, 256]] for i in format_list
        ]        
        self.topk_result(shape_format)
        
    def test_topk_shape_format_fp32_2d(self, device):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [5, 256]] for i in format_list
        ]        
        self.topk_result(shape_format)
        
    def test_topk_shape_format_fp16_3d(self, device):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [32, 8, 8]] for i in format_list
        ]        
        self.topk_result(shape_format)
        
    def test_topk_shape_format_fp32_3d(self, device):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [32, 8, 8]] for i in format_list
        ]        
        self.topk_result(shape_format)
        
    def test_topk_shape_format_fp16_4d(self, device):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [64, 112, 7, 7]] for i in format_list
        ]        
        self.topk_result(shape_format)
        
    def test_topk_shape_format_fp32_4d(self, device):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [64, 112, 7, 7]] for i in format_list
        ]        
        self.topk_result(shape_format)

instantiate_device_type_tests(TestTopK, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()