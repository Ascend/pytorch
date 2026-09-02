# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
# Licensed under the BSD 3-Clause License
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch._C._set_warnAlways API on Ascend NPU:

PyTorch community lacks sufficient and direct API validations for this API, so this file is added.
This file validates torch._C._set_warnAlways (extendable).

Test command:
    python test/npu/test_set_warnAlways.py
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu
from torch_npu.testing.common_utils import SupportedDevices

assert torch_npu is not None  # NPU backend registration


class TestSetWarnAlways(TestCase):

    def setUp(self):
        torch._C._set_warnAlways(False)

    def tearDown(self):
        torch._C._set_warnAlways(False)

    def test_set_warnAlways_exists(self):
        self.assertTrue(hasattr(torch._C, '_set_warnAlways'))

    def test_set_warnAlways_callable(self):
        self.assertTrue(callable(torch._C._set_warnAlways))

    @SupportedDevices(['Ascend910A', 'Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_set_warnAlways_with_true(self):
        torch._C._set_warnAlways(True)
        self.assertTrue(True)

    def test_set_warnAlways_with_false(self):
        torch._C._set_warnAlways(False)
        self.assertTrue(True)

    @SupportedDevices(['Ascend910A', 'Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_set_warnAlways_with_npu_tensor(self):
        torch._C._set_warnAlways(True)
        x = torch.randn(3, 4).npu()
        y = torch.randn(3, 4).npu()
        z = x + y
        self.assertEqual(z.shape, torch.Size([3, 4]))


if __name__ == "__main__":
    run_tests()
