# Copyright (c) 2026 Huawei Technologies Co., Ltd
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
"""
Add validation cases for torch.nn.parameter.is_lazy on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.nn.parameter.is_lazy (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

# Get device command to ensure tensors run on NPU
device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestIsLazy(TestCase):

    def test_is_lazy_uninitialized_parameter(self):
        """Test is_lazy returns True for UninitializedParameter"""
        uninit_param = torch.nn.parameter.UninitializedParameter(device=device_type)
        self.assertTrue(torch.nn.parameter.is_lazy(uninit_param))

    def test_is_lazy_uninitialized_buffer(self):
        """Test is_lazy returns True for UninitializedBuffer"""
        uninit_buffer = torch.nn.parameter.UninitializedBuffer(device=device_type)
        self.assertTrue(torch.nn.parameter.is_lazy(uninit_buffer))

    def test_is_lazy_regular_parameter(self):
        """Test is_lazy returns False for regular Parameter"""
        regular_param = torch.nn.Parameter(torch.randn(3, 3).to(device_type))
        self.assertFalse(torch.nn.parameter.is_lazy(regular_param))

    def test_is_lazy_regular_tensor(self):
        """Test is_lazy returns False for regular Tensor"""
        regular_tensor = torch.randn(3, 3).to(device_type)
        self.assertFalse(torch.nn.parameter.is_lazy(regular_tensor))

    def test_is_lazy_none(self):
        """Test is_lazy returns False for None"""
        self.assertFalse(torch.nn.parameter.is_lazy(None))

    def test_is_lazy_after_materialize(self):
        """Test is_lazy returns False after materialize"""
        uninit_param = torch.nn.parameter.UninitializedParameter(device=device_type)
        self.assertTrue(torch.nn.parameter.is_lazy(uninit_param))
        uninit_param.materialize(shape=(3, 3))
        self.assertFalse(torch.nn.parameter.is_lazy(uninit_param))


if __name__ == "__main__":
    run_tests()
