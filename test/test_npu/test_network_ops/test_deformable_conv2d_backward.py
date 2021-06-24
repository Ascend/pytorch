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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests

class TestDeformableConv2dBackward(TestCase):
    def create_single_npu_tensor(self, item, minvalue, maxvalue):
        dtype = item[0]
        format = item[1]
        shape = item[2]
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        npu_input = torch.from_numpy(input1).to("npu")
        if format != -1:
            npu_input = npu_input.npu_format_cast(format)
        return npu_input

    def test_deformable_conv2d(self, device):
        input = self.create_single_npu_tensor([np.float32, 0, (16, 32, 32, 32)], 0, 10)
        weight = self.create_single_npu_tensor([np.float32, 0, (32, 32, 5, 5)], 0, 10)
        offset = self.create_single_npu_tensor([np.float32, 0, (16, 75, 32, 32)], 0, 10)
        input.requires_grad = True
        weight.requires_grad = True
        offset.requires_grad = True

        npu_output, offset_out = torch.npu_deformable_conv2d(
            input, weight, offset, None, kernel_size=[5, 5], stride = [1, 1, 1, 1], padding = [2, 2, 2, 2])
        npu_output.backward(torch.ones_like(npu_output))
        npu_output = npu_output.cpu().detach()

        output = npu_output.select(1, 2).select(1, 2).select(1, 2)
        expect_output = torch.tensor([65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504.,
                                      65504., 65504., 65504., 65504., 65504., 65504., 65504.])
        self.assertRtolEqual(expect_output, output)

        input_grad = input.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_input_grad = torch.tensor([1018.5208, 1080.2323, 2533.2463, 1305.0685, 3977.8293, 2363.5681,
                                          1414.5939, 2116.5427, 1401.0662, 2064.0400, 1945.2327, 2338.5208, 
                                          300.2462, 2646.7798, 1899.1229, 2165.7280])
        self.assertRtolEqual(expect_input_grad, input_grad.cpu())

        offest_grad = offset.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_offest_grad = torch.tensor([-4708.0259, -139.2554, -2387.8149, 31017.8438, 19861.9528,
                                           -1209.2686, -24085.7285, -3950.3850, -31044.7070, 4571.3936, 
                                           582.9868, -5514.0459, 78401.6562, -1778.3700, -14311.4365,
                                           -2065.9717])
        self.assertRtolEqual(expect_offest_grad, offest_grad.cpu())

        weight_grad = weight.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_weight_grad = torch.tensor([279501.8438, 279501.8438, 279501.8438, 279501.8438, 279501.8125,
                                           279501.8125, 279501.8125, 279501.8125, 279501.8438, 279501.8438,
                                           279501.8438, 279501.8438, 279501.8438, 279501.8438, 279501.8438,
                                           279501.8438, 279501.8438, 279501.8438, 279501.8438, 279501.8438,
                                           279501.8438, 279501.8438, 279501.8438, 279501.8438, 279501.8125,
                                           279501.8125, 279501.8125, 279501.8125, 279501.8438, 279501.8438,
                                           279501.8438, 279501.8438])
        self.assertRtolEqual(expect_weight_grad, weight_grad.cpu())

instantiate_device_type_tests(TestDeformableConv2dBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
