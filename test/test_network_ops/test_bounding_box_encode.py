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
from torch_npu.testing.common_utils import create_common_tensor

class TesBoundingBoxEncode(TestCase):
    def test_encode_shape_format(self):
        input1 = torch.tensor([[1., 2., 3., 4.], [3.,4., 5., 6.]], dtype = torch.float32).to("npu")
        input2 = torch.tensor([[5., 6., 7., 8.], [7.,8., 9., 6.]], dtype = torch.float32).to("npu")
        input1_fp16 = input1.to(torch.half)
        input2_fp16 = input2.to(torch.half)
        expedt_output = torch.tensor([[13.3281, 13.3281, 0.0000, 0.0000],
                                      [13.3281,  6.6641, 0.0000, -5.4922]], dtype = torch.float32)
        output = torch_npu.npu_bounding_box_encode(input1, input2, 0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2)
        output_fp16 = torch_npu.npu_bounding_box_encode(input1_fp16, input2_fp16, 0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2)
        self.assertRtolEqual(expedt_output, output.cpu())
        self.assertRtolEqual(expedt_output.to(torch.half), output_fp16.cpu())

if __name__ == "__main__":
    run_tests()
