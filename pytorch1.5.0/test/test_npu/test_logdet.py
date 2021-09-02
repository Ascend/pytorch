
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


class TestLogDet(TestCase):

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)

        return npu_input1

    def generate_three_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)

        return npu_input1, npu_input2, npu_input3

    def generate_scalar(self, min_d, max_d):
        scalar = np.random.uniform(min_d, max_d)
        return scalar

    def generate_int_scalar(self, min_d, max_d):
        scalar = np.random.randint(min_d, max_d)
        return scalar

    def cpu_op_exec(self, input1):
        output = torch.logdet(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.logdet(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_tensor_need_to_npu(self, input1):
        input1 = input1.to("npu")
        output = torch.logdet(input1)

        output = output.to("cpu")
        output = output.numpy()
        return output

    def register_tensor(self, item,min_val,max_val):
        res = []
        cpu_input, npu_input = create_common_tensor(item, min_val , max_val)
        det_result = torch.det(cpu_input)
        for i in range(len(det_result)):
            if det_result[i] > 0:
                res.append(cpu_input[i])
        res = torch.stack(res)
        return res, len(res)

    def register_tensor_fp16(self, item,min_val,max_val):
        res = []
        cpu_input, npu_input = create_common_tensor(item,  min_val , max_val)
        cpu_input_tmp = cpu_input.to(torch.float32)
        det_result = torch.det(cpu_input_tmp)
        for i in range(len(det_result)):
            if det_result[i] > 0:
                res.append(cpu_input[i])
        res = torch.stack(res)
        return res, len(res)

    def create_det_tensor(self, input_tensor):
        cpu_input = input_tensor
        npu_input = input_tensor.to("npu")
        return cpu_input, npu_input

    def test_logdet_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (6, 2, 2)], 100, 200],
            [[np.float32, -1, (24, 5, 5)], 100, 200],
            [[np.float32, -1, (14, 5, 5)], 21, 22],
            [[np.float32, -1, (74,4,4)], 21205, 22225],
            [[np.float32, -1, (58,4,4)], -30,30],
            [[np.float32, -1, (30,16,16)], -30,30],
            [[np.float32, -1, (58, 4, 4)], 0.3219780311757745 , 92],
            [[np.float32, -1, (32, 16, 16)], 0.4820305734500543 , 28],
            [[np.float32, -1, (28, 8, 8)], 0.8563874665918477 , 98],
            [[np.float32, -1, (42, 6, 6)], 0.0694198357720135 , 50],
            [[np.float32, -1, (12, 10, 10)], 0.3316939248453338 , 17],
            [[np.float32, -1, (6, 10, 10)], 0.6447298684351989 , 95],
            [[np.float32, -1, (6, 9, 9)],0.8723538084975545 , 85],
            [[np.float32, -1, (10, 5, 5)], 0.8283759153463854 , 71],
            [[np.float32, -1, (10, 1, 1)], 0.24718684227306953 , 1],
            [[np.float32, -1, (6,1,1)], 0.0694198357720135, 0.24718684227306953],
            [[np.float32, -1, (8, 10, 10)], 0.7866457165672994 , 5],
            [[np.float32, -1, (6, 14, 14)], 0.9956475043306917 , 28],
            [[np.float32, -1, (6, 7, 7)],0.3793216987112159 , 39],
            [[np.float32, -1, (14, 10, 10)], 0.769565434387681 , 9],
            [[np.float32, -1, (16, 10, 10)], 0.8039978883789274 , 22],
            [[np.float32, -1, (30, 3, 3)], 0.03133650248813469 , 37],
            [[np.float32, -1, (4, 1, 1)], 0.853775978441379 , 34 ],
            [[np.float32, -1, (18, 6, 6)], 0.503285855595573 , 35],
            [[np.float32, -1, (6, 3, 3)], 1, 10],
        ]
        for item in shape_format:
            input_shape = item[0][2]
            res, tmp_shape0 = self.register_tensor(item[0],item[1],item[2])
            cpu_input1, npu_input1 = self.create_det_tensor(res)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            input_shape = list(input_shape)
            input_shape[0] = tmp_shape0
            self.assertRtolEqual(cpu_output, npu_output)

    def test_logdet_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1):
            input1 = input1.to(torch.float32)
            output = torch.logdet(input1)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (9, 5, 5)],-2,2],
            [[np.float16, -1, (60,4,4)],-10,12],
            [[np.float16, -1, (12,5,5)], 5,10],
            [[np.float16, -1, (14, 5, 5)], 0.9283381566708346 , 10],
            [[np.float16, -1, (71, 2, 2)], 0.6234465730020081 , 13],
            [[np.float16, -1, (10, 5, 5)], 0.7440899332166594 , 1],
            [[np.float16, -1, (13, 5, 5)], 0.9790231845699171 , 9],
            [[np.float16, -1, (10, 7, 7)], 0.7852605507867441 , 8],
            [[np.float16, -1, (18, 2, 2)], 0.8758750778305631 , 9],
            [[np.float16, -1, (10, 6, 6)], 0.7570129172808612 , 5],
            [[np.float16, -1, (7, 7, 7)], 0 , 2],
            [[np.float16, -1, (9, 5, 5)], 1 , 2],
            [[np.float16, -1, (12, 4, 4)], 0.7349293532899402 , 19],
            [[np.float16, -1, (15, 8, 8)], 0.9583309378850908 , 3],
            [[np.float16, -1, (11, 2, 2)],0.3560076034004038 , 25],
        ]

        for item in shape_format:
            input_shape = item[0][2]
            res, tmp_shape0 = self.register_tensor_fp16(item[0],item[1],item[2])
            cpu_input1,npu_input1 = self.create_det_tensor(res)
            cpu_output = cpu_op_exec_fp16(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            input_shape = list(input_shape)
            input_shape[0] = tmp_shape0
            self.assertRtolEqual(cpu_output, npu_output,prec=1e-3)


instantiate_device_type_tests(TestLogDet, globals(), except_for='cpu')

if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()
