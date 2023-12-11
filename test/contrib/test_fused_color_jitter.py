# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

from PIL import Image
import numpy as np

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.module import FusedColorJitter


class TestFusedColorJitter(TestCase):
    def test_fusedcolorjitter(self):
        image = Image.fromarray(torch.randint(
            0, 256, size=(224, 224, 3)).numpy().astype(np.uint8))
        fcj = FusedColorJitter(0.1, 0.1, 0.1, 0.1)
        output = fcj(image)
        self.assertEqual(output is not None, True)


if __name__ == "__main__":
    run_tests()
