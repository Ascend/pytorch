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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

class TestUniqueConsecutive(TestCase):
    def test_unique_consecutive(self, device="npu"):
        shape_format = [
                        [[torch.int32, (2, 3)], 0],
                        [[torch.long, (2, 3)], 1],
                        [[torch.float32, (2, 3)], 0],
                        [[torch.float16, (2, 3)], 1],
                        [[torch.int32, (2, 3)], None],
                        [[torch.long, (2, 3)], None],
                        [[torch.float32, (2, 3)], None],
                        [[torch.float16, (2, 3)], None]
        ]

        for item in shape_format:
            cpu_input = torch.rand(item[0][1]).random_(0, 3).to(item[0][0])
            npu_input = cpu_input.npu()
            if item[0][0] == torch.float16:
                cpu_input = cpu_input.float()

            cpu_output, cpu_idx, cpu_counts = torch.unique_consecutive(cpu_input, return_inverse=True, 
                                                                       return_counts=True, dim=item[1])
            npu_output, npu_idx, npu_counts = torch.unique_consecutive(npu_input, return_inverse=True, 
                                                                       return_counts=True, dim=item[1])

            if item[0][0] == torch.float16:
                cpu_output = cpu_output.half()
            self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())
            self.assertRtolEqual(cpu_idx.numpy(), npu_idx.cpu().numpy())
            self.assertRtolEqual(cpu_counts.numpy(), npu_counts.cpu().numpy())
    
    def test_unique_consecutive_case_in_dino(self, device="npu"):
        input_list = [
            torch.tensor([224, 224, 96, 96, 96, 96, 96, 96, 96, 96]),
            torch.tensor([224, 224])
        ]
        for i in input_list:
            cpu_output, cpu_counts = torch.unique_consecutive(i, return_counts=True)
            npu_output, npu_counts = torch.unique_consecutive(i.npu(), return_counts=True)
            self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())
            self.assertRtolEqual(cpu_counts.numpy(), npu_counts.cpu().numpy())

if __name__ == "__main__":
    run_tests() 