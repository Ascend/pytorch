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
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestMeshgrid(TestCase):
    def test_meshgrid(self, device):
        a = torch.tensor(1, dtype=torch.int32).npu()
        b = torch.tensor([1, 2, 3], dtype=torch.int32).npu()
        c = torch.tensor([1, 2], dtype=torch.int32).npu()
        grid_a, grid_b, grid_c = torch.meshgrid([a, b, c])
        self.assertEqual(grid_a.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c.shape, torch.Size([1, 3, 2]))
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c)
        self.assertEqual(grid_a2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c2.shape, torch.Size([1, 3, 2]))
        expected_grid_a = torch.ones(1, 3, 2, dtype=torch.int32).npu()
        expected_grid_b = torch.tensor([[[1, 1],
                                         [2, 2],
                                         [3, 3]]], dtype=torch.int32).npu()
        expected_grid_c = torch.tensor([[[1, 2],
                                         [1, 2],
                                         [1, 2]]], dtype=torch.int32).npu()
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))
    
    def cpu_op_exec(self, input1, input2):
        output1, output2 = torch.meshgrid(input1, input2)
        output1 = output1.numpy()
        output2 = output2.numpy()
        return output1, output2
    
    def npu_op_exec(self, input1, input2):
        output1, output2 = torch.meshgrid(input1, input2)
        output1 = output1.to("cpu")
        output2 = output2.to("cpu")
        output1 = output1.numpy()
        output2 = output2.numpy()
        return output1, output2
    
    def test_meshgrid_case(self, device):
        case = [
                [torch.tensor([1, 2, 3], dtype=torch.int32), torch.tensor([4, 5, 6], dtype=torch.int32)]
        ]

        for item in case:
            cpu_output1, cpu_output2 = self.cpu_op_exec(item[0], item[1])
            npu_output1, npu_output2 = self.npu_op_exec(item[0].npu(), item[1].npu())
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)


instantiate_device_type_tests(TestMeshgrid, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
    

