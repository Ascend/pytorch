# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
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

import torch
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from graph_utils import RunFuncInGraphMode


class TestDiv(TestCase):
    '''
    @dtypes(torch.float)
    @RunFuncInGraphMode
    def test_div(self, device, dtype):
        m1 = torch.randn(10, 10, dtype=torch.float, device="cpu").to(dtype=dtype)
        m1 = m1.to("npu")
        res1 = m1.clone()
        res1[:, 3].div_(2)
        res2 = m1.clone()
        for i in range(m1.size(0)):
            res2[i, 3] = res2[i, 3] / 2
        self.assertEqual(res1.to("cpu"), res2.to("cpu"))

        if dtype == torch.bfloat16:
            a1 = torch.tensor([4.2, 6.2], dtype=dtype, device=device)
            a2 = torch.tensor([2., 2.], dtype=dtype, device=device)
            self.assertEqual((a1 / a2).to("cpu"),
                             torch.tensor([2.1, 3.1], dtype=dtype, device=device).to("cpu"),
                             0.01)
            self.assertEqual(a1.div(a2).to("cpu"), (a1 / a2).to("cpu"))
    '''

    def cpu_op_exec(self, input1, input2):
        output = torch.div(input1, input2)
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.div(input1, input2)
        output = output.to("cpu")
        return output

    @RunFuncInGraphMode
    def test_div_shape_format(self, device):
        shape_format = [
                [[np.float32, 0, 1], [np.float32, 0, 1]],
                [[np.float32, 0, (64, 10)], [np.float32, 0, 1]],
                [[np.float32, 3, (256, 2048, 7, 7)], [np.float32, 0, 1]],
                [[np.float32, 4, (32, 1, 3, 3)], [np.float32, 4, (32, 1, 3, 3)]],
                [[np.float32, 29, (10, 128)], [np.float32, 29, (10, 128)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestDiv, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
