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


class TestPixel_shuffle(TestCase):


    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1


    def cpu_op_exec(self, input1, block_size):
        output = torch.nn.functional.pixel_shuffle(input1, block_size)
        output = output.numpy()
        return output

    def npu_op_exec_tensor_need_to_npu(self, input1, block_size):
        input1 = input1.to("npu")
        output = torch.nn.functional.pixel_shuffle(input1, block_size)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_pixel_shuffle_common_shape_format(self, device):
        shape_format = [
            [np.float32, -1, (1, 16, 4, 4)],
            [np.float32, -1, (1, 16, 2, 2)],
            [np.float32, -1, (1, 16, 1, 1)],
            [np.float32, -1, (1, 64, 1, 1)],
            [np.float32, -1, (1, 256, 1, 1)],
            [np.float32, -1, (1, 655360, 1, 1)],
            #[np.int8, -1, (1, 786432, 1, 1)],
            #[np.int64, -1, (1, 655360, 1, 1)],
            #[np.uint8, -1, (1, 655360, 1, 1)],
            [np.int32, -1, (1, 655360, 1, 1)]        
           ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, 4)
            npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, 4)
            self.assertRtolEqual(cpu_output, npu_output)


        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, 1)
            npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, 1)
            self.assertRtolEqual(cpu_output, npu_output)


    def test_pixel_shuffle_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1, block_size):
            input1 = input1.to(torch.float32)
            output = torch.pixel_shuffle(input1, block_size)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [np.float16, -1, (1, 16, 1, 1)],
            [np.float16, -1, (1, 16, 4, 4)],
            [np.float16, -1, (1, 655360, 1, 1)]
      ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input1, 4)
            npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, 4)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestPixel_shuffle, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:7")
    run_tests()
