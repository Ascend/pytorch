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


import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMaskedScatter(TestCase):

    def cpu_op_exec(self, input1, maskbool, source):
        cpu_output = torch.masked_scatter(input1, maskbool, source)
        return cpu_output.numpy()

    def npu_op_exec(self, input1, maskbool, source):
        input1 = input1.to("npu")
        maskbool = maskbool.to("npu")
        source = source.to("npu")
        npu_output = torch.masked_scatter(input1, maskbool, source)
        npu_output = npu_output.to("cpu")
        return npu_output.numpy()

    def cpu_inp_op_exec(self, input1, maskbool, source):
        cpu_output = input1.masked_scatter_(maskbool, source)
        return cpu_output.numpy()

    def npu_inp_op_exec(self, input1, maskbool, source):
        maskbool = maskbool.to("npu")
        npu_output = input1.masked_scatter_(maskbool, source)
        npu_output = npu_output.to("cpu")
        return npu_output.numpy()

    def test_masked_scatter_float(self):
        dtype_list = [np.float32, np.float16]
        format_list = [0, 3]
        shape_list = [[4, 5], [3, 4, 5], [2, 3, 4, 5]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        mask = torch.randn(4, 1)
        maskbool = mask.ge(0.5)

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_source, npu_source = create_common_tensor(item, 0, 100)
            cpu_dtype = cpu_input.dtype
            if cpu_dtype == torch.float16:
                cpu_input = cpu_input.float()
                cpu_source = cpu_source.float()
            cpu_output2 = self.cpu_inp_op_exec(cpu_input, maskbool, cpu_source)
            npu_output2 = self.npu_inp_op_exec(npu_input, maskbool, npu_source)
            if cpu_dtype == torch.float16:
                cpu_output2 = cpu_output2.astype(np.float16)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_masked_scatter_int(self):
        dtype_list = [np.int32, np.int64]
        format_list = [0]
        shape_list = [[4, 5], [3, 4, 5], [2, 3, 4, 5]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        mask = torch.randn(4, 1)
        maskbool = mask.ge(0.5)

        for item in shape_format:
            cpu_source, npu_source = create_common_tensor(item, 0, 100)
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output2 = self.cpu_inp_op_exec(cpu_input, maskbool, cpu_source)
            npu_output2 = self.npu_inp_op_exec(npu_input, maskbool, npu_source)
            self.assertRtolEqual(cpu_output2, npu_output2)


if __name__ == "__main__":
    run_tests()
