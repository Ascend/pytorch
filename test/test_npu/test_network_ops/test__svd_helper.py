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
import torch.nn.functional as F
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import instantiate_device_type_tests
from util_test import create_common_tensor

class TestSvdHelper(TestCase):
    def cpu_op_exec(self, input1, some, compute_uv=False):
        output_u, output_s, output_v = torch.svd(input1, some, compute_uv)
        return output_u, output_s, output_v

    def npu_op_exec(self, input1, some, compute_uv=False):
        output_u, output_s, output_v = torch.svd(input1, some, compute_uv)
        output_u = output_u.cpu()
        output_s = output_s.cpu()
        output_v = output_v.cpu()
        return output_u, output_s, output_v

    def test_svd_fp32(self, device):
        shape_format = [
            [[np.float32, -1, [5, 3]]],
            [[np.float32, -1, [2, 3, 4]]],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)

            cpu_u, cpu_s, cpu_v = self.cpu_op_exec(cpu_input, some=True)
            npu_u, npu_s, npu_v = self.npu_op_exec(npu_input, some=True)
            self.assertRtolEqual(cpu_u, npu_u)
            self.assertRtolEqual(cpu_s, npu_s)
            self.assertRtolEqual(cpu_v, npu_v)

            cpu_u, cpu_s, cpu_v = self.cpu_op_exec(cpu_input, some=False)
            npu_u, npu_s, npu_v = self.npu_op_exec(npu_input, some=False)
            self.assertRtolEqual(cpu_u, npu_u)
            self.assertRtolEqual(cpu_s, npu_s)
            self.assertRtolEqual(cpu_v, npu_v)



instantiate_device_type_tests(TestSvdHelper, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
