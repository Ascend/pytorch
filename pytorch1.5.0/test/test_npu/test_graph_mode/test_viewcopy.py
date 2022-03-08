# Copyright (c) 2022, Huawei Technologies.All rights reserved.
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
from graph_utils import graph_mode

class TestViewCopy(TestCase):
    def inplce_op_exec_1(self, input1, input2):
        input1[:, :, :1].copy_(input2[:, :, :1])

    def inplce_op_exec_2(self, input1, input2):
        input1[:, :, 1:3].copy_(input2[:, :, 1:3])

    def inplce_op_exec_3(self, input1, input2):
        input1[:3, :, :].copy_(input2[:3, :, :])

    def inplce_op_exec_4(self, input1, input2):
        input1[2:4, :, :].copy_(input2[2:4, :, :])

    def inplce_op_exec_5(self, input1, input2):
        input1.add_(input2)
        tmp = input2 * 2 - input2[:1]
        input1[:, :2, 1:3, :].copy_(tmp[:, :2, 1:3, :])
        res = input1 * 2 + 1
        return res

    def inplce_op_exec_6(self, input1, input2):
        input1[0:2, :, 1:2, :].copy_(input2[0:2, :, 1:2, :1])
        res = input1 + input1
        return res

    def inplce_op_exec_7(self, input1, input2):
        input1[:, :, :, :2].copy_(input2[1:2, :, :, 1:2])
        res = input1 + input1
        return res


    @graph_mode
    def test_viewcopy_last_dim(self, device):
        dtype_list = [np.uint8, np.int8, np.int16, np.int32, np.float32, np.double]
        format_list = [-1]
        shape_list = [(5, 6, 7)]
        dtype_shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in dtype_shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)

            self.inplce_op_exec_1(cpu_input1, cpu_input2)
            self.inplce_op_exec_1(npu_input1, npu_input2)
            npu_input1_tocpu = npu_input1.cpu()
            self.assertRtolEqual(cpu_input1, npu_input1_tocpu)
            
            self.inplce_op_exec_2(cpu_input1, cpu_input2)
            self.inplce_op_exec_2(npu_input1, npu_input2)
            npu_input1_tocpu = npu_input1.cpu()
            self.assertRtolEqual(cpu_input1, npu_input1_tocpu)

    @graph_mode
    def test_viewcopy_first_dim(self, device):
        dtype_list = [np.uint8, np.int8, np.int16, np.int32, np.float32, np.double]
        format_list = [-1]
        shape_list = [(5, 6, 7)]
        dtype_shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in dtype_shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)

            self.inplce_op_exec_3(cpu_input1, cpu_input2)
            self.inplce_op_exec_3(npu_input1, npu_input2)
            npu_input1_tocpu = npu_input1.cpu()
            self.assertRtolEqual(cpu_input1, npu_input1_tocpu)

            self.inplce_op_exec_4(cpu_input1, cpu_input2)
            self.inplce_op_exec_4(npu_input1, npu_input2)
            npu_input1_tocpu = npu_input1.cpu()
            self.assertRtolEqual(cpu_input1, npu_input1_tocpu)


    @graph_mode
    def test_viewcopy_compute(self, device):
        dtype_list = [np.float32]
        format_list = [-1]
        shape_list = [(2, 4, 5, 3)]
        dtype_shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in dtype_shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)

            cpu_res = self.inplce_op_exec_5(cpu_input1, cpu_input2)
            npu_res = self.inplce_op_exec_5(npu_input1, npu_input2)
            npu_res = npu_res.cpu()
            self.assertRtolEqual(cpu_res, npu_res)

    @graph_mode
    def test_viewcopy_broadcast(self, device):
        dtype_list = [np.float32]
        format_list = [-1]
        shape_list = [(4, 1, 2, 3)]
        dtype_shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in dtype_shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 10)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 10)

            cpu_res = self.inplce_op_exec_6(cpu_input1, cpu_input2)
            npu_res = self.inplce_op_exec_6(npu_input1, npu_input2)
            npu_res = npu_res.cpu()
            self.assertRtolEqual(cpu_res, npu_res)

            cpu_res = self.inplce_op_exec_7(cpu_input1, cpu_input2)
            npu_res = self.inplce_op_exec_7(npu_input1, npu_input2)
            npu_res = npu_res.cpu()
            self.assertRtolEqual(cpu_res, npu_res)

instantiate_device_type_tests(TestViewCopy, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()

