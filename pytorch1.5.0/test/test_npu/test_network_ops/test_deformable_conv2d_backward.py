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

    def test_deformable_conv2d_backward_fp32(self, device):
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

        offset_out_select = offset_out.select(1, 2).select(1, 2).select(1, 2)
        expect_offset_out = torch.tensor([13.6374, 6.0295, 79.6111, 33.5996, 53.6248, 62.2289, 14.3497, 47.6463,
                                          52.3312, 34.1246, 8.6705, 3.3515, 9.9513, 15.3604, 38.9772, 57.1306])
        self.assertRtolEqual(expect_offset_out, offset_out_select.cpu().detach())

        input_grad = input.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_input_grad = torch.tensor([1018.3082, 1080.2413, 2533.7673, 1305.1088, 3977.8022, 2363.5249,
                                          1414.8381, 2117.0735, 1400.9083, 2064.4629, 1945.1212, 2338.8213, 
                                          300.2924, 2646.9910, 1898.9320, 2165.8921])
        self.assertRtolEqual(expect_input_grad, input_grad.cpu())

        offest_grad = offset.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_offest_grad = torch.tensor([-4707.2891, -139.2936, -2391.8394, 31024.4375, 19856.1621,
                                           -1205.9329, -24091.1953, -3947.4133, -31050.9805, 4570.9854, 
                                           582.9227, -5515.2178, 78396.9062, -1778.6783, -14314.9893,
                                           -2066.9614])
        self.assertRtolEqual(expect_offest_grad, offest_grad.cpu())

        weight_grad = weight.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_weight_grad = torch.tensor([279501.8438, 279501.8438, 279501.8438, 279501.8438, 279501.8438,
                                           279501.8438, 279501.8438, 279501.8438, 279501.8750, 279501.8750,
                                           279501.8750, 279501.8750, 279501.8750, 279501.8750, 279501.8750,
                                           279501.8750, 279501.8438, 279501.8438, 279501.8438, 279501.8438,
                                           279501.8125, 279501.8125, 279501.8125, 279501.8125, 279501.8750,
                                           279501.8750, 279501.8750, 279501.8750, 279501.8750, 279501.8750,
                                           279501.8750, 279501.8750])
        self.assertRtolEqual(expect_weight_grad, weight_grad.cpu())

    def test_deformable_conv2d_backward_fp16(self, device):
        input_fp16 = self.create_single_npu_tensor([np.float16, 0, (16, 32, 32, 32)], 0, 10)
        weight = self.create_single_npu_tensor([np.float16, 0, (32, 32, 5, 5)], 0, 10)
        offset = self.create_single_npu_tensor([np.float16, 0, (16, 75, 32, 32)], 0, 10)
        input_fp16.requires_grad = True
        weight.requires_grad = True
        offset.requires_grad = True

        npu_output, offset_out = torch.npu_deformable_conv2d(
            input_fp16, weight, offset, None, kernel_size=[5, 5], stride = [1, 1, 1, 1], padding = [2, 2, 2, 2])
        npu_output.backward(torch.ones_like(npu_output))
        npu_output = npu_output.cpu().detach()

        output = npu_output.select(1, 2).select(1, 2).select(1, 2)
        expect_output = torch.tensor([65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504.,
                                      65504., 65504., 65504., 65504., 65504., 65504., 65504.], dtype=torch.float16)
        self.assertRtolEqual(expect_output, output)

        offset_out_select = offset_out.select(1, 2).select(1, 2).select(1, 2)
        expect_offset_out = torch.tensor([13.6562, 6.0352, 79.4375, 33.5938, 53.6875, 62.2188, 14.3438, 47.6562,
                                          52.3750, 34.0938, 8.6797, 3.3516, 9.9531, 15.3750, 39.0625, 57.1875], 
                                          dtype=torch.float16)
        self.assertRtolEqual(expect_offset_out, offset_out_select.cpu().detach())

        input_grad = input_fp16.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_input_grad = torch.tensor([1019.0000, 1082.0000, 2534.0000, 1304.0000, 3978.0000, 2364.0000,
                                          1413.0000, 2114.0000, 1402.0000, 2064.0000, 1944.0000, 2340.0000, 
                                          299.7500, 2648.0000, 1895.0000, 2164.0000], dtype=torch.float16)
        self.assertRtolEqual(expect_input_grad, input_grad.cpu())

        offest_grad = offset.grad.select(1, 2).select(1, 2).select(1, 2)
        expect_offest_grad = torch.tensor([141.2500, 34784.0000, -12384.0000, -1885.0000, -5440.0000,
                                           4416.0000, 13920.0000, -19952.0000, 4160.0000, 24848.0000, 
                                           -1464.0000, -21088.0000, -1060.0000, -22544.0000, 9152.0000,
                                           -4312.0000], dtype=torch.float16)
        self.assertRtolEqual(expect_offest_grad, offest_grad.cpu())

        weight_grad = weight.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_weight_grad = torch.tensor([65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 
                                           65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 
                                           65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 
                                           65504., 65504., 65504., 65504., 65504.], dtype=torch.float16)
        self.assertRtolEqual(expect_weight_grad, weight_grad.cpu())

instantiate_device_type_tests(TestDeformableConv2dBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
