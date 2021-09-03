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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestMaskedScale(TestCase):
    def generate_data(self, dtype, shape, min_d, max_d):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        cpu_input = input1
        npu_input = torch.from_numpy(input1).to("npu")
        return cpu_input, npu_input

    def generate_mask(self, shape):
        mask = torch.empty(shape,dtype=torch.int8).random_(2)
        cpu_mask = mask.numpy()
        return cpu_mask, mask

    def dynamic_generate_data(self, data_type):
        format_list = []
        shape_range = [2,5]
        min_value_range = [-100, 0, 1]
        max_value_range = [100, 1, 1000]
        for shape in shape_range:
            for min_v, max_v in zip(min_value_range, max_value_range):
                shape_v = [np.random.randint(1, 50) for _ in range(shape)]
                format_list.append(
                    [data_type, shape_v, min_v, max_v]
                )
        return format_list

    def numpy_op_exec_masked_scale(self, input1, mask, value):
        res = input1 * mask * value
        return res

    def npu_op_exec_masked_scale(self, input1, mask, value):
        input1 = input1.npu()
        mask = mask.npu()
        value = torch.tensor(value)
        res = torch._masked_scale(input1, mask, value)
        res = res.to("cpu")
        res = res.detach().numpy()
        return res

    def test_masked_scale_format_fp16(self,device):
        self._test_masked_scale_format(device, np.float16)

    def test_masked_scale_format_fp32(self,device):
        self._test_masked_scale_format(device, np.float32)

    def _test_masked_scale_format(self, device, dtype):
        format_list = self.dynamic_generate_data(dtype)
        for item in format_list:
            cpu_input, npu_input = self.generate_data(*item)
            cpu_mask, npu_mask = self.generate_mask(item[1])
            scale = np.random.uniform(0,1)
            cpu_output = self.numpy_op_exec_masked_scale(cpu_input,cpu_mask,scale)
            npu_output = self.npu_op_exec_masked_scale(npu_input,npu_mask,scale)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestMaskedScale, globals(), except_for='cpu')     
if __name__ == '__main__': 
    torch.npu.set_device("npu:0")
    run_tests()
