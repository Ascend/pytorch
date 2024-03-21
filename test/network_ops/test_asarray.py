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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAsarray(TestCase):
    def test_asarray_default(self):
        np_input = np.random.randn(1)
        cpu_output_default = torch.asarray(np_input, device="cpu").numpy()
        npu_output_default = torch.asarray(np_input, device="npu").cpu().numpy()
        self.assertRtolEqual(cpu_output_default, npu_output_default)

    def test_asarray_device_none(self):
        npu_input = torch.tensor([1, 2, 3]).npu()
        npu_input_device = npu_input.device
        npu_output = torch.asarray(npu_input)
        npu_output_device = npu_output.device
        assert npu_input_device == npu_output_device

if __name__ == "__main__":
    run_tests()
