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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestAll(TestCase):
    def create_bool_tensor(self, shape, minValue, maxValue):
        input1 = np.random.uniform(minValue, maxValue, shape)
        input1 = input1 > 0.5
        cpu_input = torch.from_numpy(input1)
        npu_input = torch.from_numpy(input1).to("npu")
        return cpu_input, npu_input

    def cpu_op_exec(self, input1):
        output = input1.all()
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = input1.all()
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_all_shape_format(self):
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1024], [2, 0, 2]]
        for item in shape_list:
            cpu_input, npu_input = self.create_bool_tensor(item, 0, 1)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(
                cpu_output.astype(
                    np.int32), npu_output.astype(
                    np.int32))

    def cpu_op_exec1(self, input1, dim):
        output = input1.all(dim=dim)
        output = output.numpy()
        return output

    def npu_op_exec1(self, input1, dim):
        output = input1.all(dim=dim)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def npu_op_out_exec1(self, input1, dim):
        shape = list(input1.shape)
        output0 = torch.randn(shape) > 0
        output1 = torch.randn(shape.pop()) > 0
        output0 = output0.npu()
        output1 = output1.npu()
        torch.all(input1, dim=dim, keepdim = False, out = output0)
        torch.all(input1, dim=dim, keepdim = False, out = output1)
        output0 = output0.to("cpu").numpy()
        output1 = output1.to("cpu").numpy()
        return output0, output1

    def test_alld_shape_format(self):
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1024]]
        for item in shape_list:
            cpu_input, npu_input = self.create_bool_tensor(item, 0, 1)
            cpu_output = self.cpu_op_exec1(cpu_input, 0)
            npu_output = self.npu_op_exec1(npu_input, 0)
            npu_out0, npu_out1 = self.npu_op_out_exec1(npu_input, 0)
            self.assertRtolEqual(cpu_output.astype(np.int32), npu_output.astype(np.int32))
            self.assertRtolEqual(cpu_output.astype(np.int32), npu_out0.astype(np.int32))
            self.assertRtolEqual(cpu_output.astype(np.int32), npu_out1.astype(np.int32))

    def test_all_tensor_numel_0(self):
        ca = torch.rand(1, 2, 0, 3, 4).bool()
        na = ca.npu()
        cout = ca.all(2)
        nout = na.all(2)
        cout = cout.numpy()
        nout = nout.to("cpu").numpy()
        self.assertRtolEqual(cout, nout)


if __name__ == "__main__":
    run_tests()