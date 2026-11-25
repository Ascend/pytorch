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
Add validation cases for torch.onnx APIs on NPU:

1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.onnx.operators.shape_as_tensor (extendable).
"""

import torch
import torch.onnx.operators
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestOnnxOperators(TestCase):

    def test_shape_as_tensor_shapes(self):
        test_cases = (
            (torch.tensor(1).to(device_type), []),
            (torch.zeros(5).to(device_type), [5]),
            (torch.zeros(2, 3).to(device_type), [2, 3]),
            (torch.zeros(2, 0, 4).to(device_type), [2, 0, 4]),
        )

        for input_tensor, expected_shape in test_cases:
            actual = torch.onnx.operators.shape_as_tensor(input_tensor)
            expected = torch.tensor(expected_shape, dtype=torch.int64)
            self.assertEqual(actual, expected)
            self.assertEqual(actual.dtype, torch.int64)
            self.assertEqual(actual.device.type, "cpu")

    def test_shape_as_tensor_input_dtypes(self):
        for dtype in (torch.bool, torch.int32, torch.float32):
            input_tensor = torch.zeros(2, 3, dtype=dtype).to(device_type)
            actual = torch.onnx.operators.shape_as_tensor(input_tensor)
            self.assertEqual(actual, torch.tensor([2, 3], dtype=torch.int64))

    def test_shape_as_tensor_invalid_inputs(self):
        for invalid_input in (None, [1, 2], 1):
            with self.assertRaises(TypeError):
                torch.onnx.operators.shape_as_tensor(invalid_input)


if __name__ == "__main__":
    run_tests()
