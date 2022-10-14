# Copyright (c) 2022, Huawei Technologies.All rights reserved.
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
from torch_npu.testing.common_utils import create_common_tensor


class TestCropAndResize(TestCase):
    def result_error(self, npu_output, boxes, crop_size):
        if npu_output.dtype != boxes.dtype:
            self.fail("dtype error")
        if npu_output.shape[0] != boxes.shape[0] or npu_output.shape[2] != crop_size[0] \
            or npu_output.shape[3] != crop_size[1]:
            self.fail("shape error")

    def test_crop_and_resize(self, device="npu"):
        input1 = np.random.uniform(0, 255, (1, 3, 224, 224)).astype(np.uint8)
        npu_input1 = torch.from_numpy(input1).npu()
        boxes = torch.tensor([[0.3, 0, 1, 1], [0.2, 0.6, 1.3, 0.9]], dtype=torch.float32).npu()
        box_index = torch.tensor([0, 0], dtype=torch.int32).npu()
        crop_size = [200, 100]
        npu_output = torch_npu.crop_and_resize(npu_input1,
                                               boxes=boxes, box_index=box_index,
                                               crop_size=crop_size, method="nearest")

        self.result_error(npu_output, boxes, crop_size)


if __name__ == "__main__":
    run_tests()