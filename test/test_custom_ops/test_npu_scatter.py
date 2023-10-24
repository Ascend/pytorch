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
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuScatter(TestCase):
    def supported_op_exec(self, input1, indices, updates, dim):
        tmp = input1.reshape(-1)
        shape = input1.shape
        dim_len = shape[dim]

        for i in range(indices.numel()):
            tmp[i * dim_len + indices[i]] = updates[i]

        output = tmp.reshape(shape).to('cpu')
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, indices, updates, dim):
        output = torch_npu.npu_scatter(input1, indices, updates, dim)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_npu_scatter(self, device="npu"):
        input1_list = [[[1.6279, 0.1226], [0.9041, 1.0980]]]
        indices_list = [[0, 1]]
        updates_list = [[-1.1993, -1.5247]]
        dim_list = [0]
        exoutput_list = [[[-1.1993, 0.1226], [0.9041, -1.5247]]]

        shape_format = [[i, j, k, h, f] for i in input1_list
                        for j in indices_list for k in updates_list for h in dim_list for f in exoutput_list]

        for item in shape_format:
            input1_tensor = torch.tensor(item[0]).npu()
            indices_tensor = torch.tensor(item[1]).npu().to(torch.int32)
            updates_tensor = torch.tensor(item[2]).npu()
            dim = item[3]
            exoutput_tensor = torch.tensor(item[4])
            output1 = self.npu_op_exec(input1_tensor, indices_tensor, updates_tensor, dim)
            output2 = self.supported_op_exec(input1_tensor, indices_tensor, updates_tensor, dim)
            self.assertRtolEqual(exoutput_tensor.numpy(), output1)
            self.assertRtolEqual(output1, output2)


if __name__ == "__main__":
    run_tests()
