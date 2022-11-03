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


class TestImageNormalize(TestCase):
    def result_error(self, npu_input, npu_output):
        if npu_output.shape != npu_input.shape:
            self.fail("shape error")
        if npu_output.dtype != npu_input.dtype:
            self.fail("dtype error")

    def test_image_normalize(self, device="npu"):
        input = np.random.uniform(0, 1, (1, 3, 224, 224)).astype(np.float32)
        npu_input = torch.from_numpy(input).npu()

        mean = torch.tensor([0.485, 0.456, 0.406]).npu()
        variance = torch.tensor([0.229, 0.224, 0.225]).npu()

        npu_output = torch_npu.image_normalize(npu_input, mean, variance, dtype = 0)

        self.result_error(npu_input, npu_output)


if __name__ == "__main__":
    if torch_npu.npu.get_device_name(0)[:10] == 'Ascend910B':
        run_tests()