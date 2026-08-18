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
Add validation cases for torch.onnx.OnnxRegistry on NPU:
1. PyTorch community has this API only in v2.7 (deprecated since 2.7 and removed
   in later versions), and lacks direct test validations, so this file is added.
2. This file validates the ONNX operator registry used by the dynamo exporter:
   construction, opset version, op lookup and custom op registration (extendable).
"""

import onnxscript
from onnxscript import opset17 as op

import torch
from torch.onnx import OnnxRegistry
from torch.testing._internal.common_utils import run_tests, TestCase


@onnxscript.script(default_opset=op)
def custom_identity(x):
    return op.Identity(x)


class TestOnnxRegistry(TestCase):
    """Tests for torch.onnx.OnnxRegistry"""

    def test_registry_initialized_from_torchlib(self):
        registry = OnnxRegistry()
        self.assertTrue(registry.is_registered_op("aten", "relu"))

    def test_opset_version_is_positive_int(self):
        registry = OnnxRegistry()
        self.assertIsInstance(registry.opset_version, int)
        self.assertGreater(registry.opset_version, 0)

    def test_get_op_functions_for_registered_op(self):
        registry = OnnxRegistry()
        functions = registry.get_op_functions("aten", "matmul")
        self.assertIsNotNone(functions)
        self.assertGreater(len(functions), 0)

    def test_unknown_op_returns_none(self):
        registry = OnnxRegistry()
        self.assertIsNone(registry.get_op_functions("nonexistent_ns", "no_such_op"))
        self.assertFalse(registry.is_registered_op("nonexistent_ns", "no_such_op"))

    def test_register_custom_op(self):
        registry = OnnxRegistry()
        self.assertFalse(registry.is_registered_op("custom_ops", "my_identity"))
        registry.register_op(custom_identity, "custom_ops", "my_identity")
        self.assertTrue(registry.is_registered_op("custom_ops", "my_identity"))
        functions = registry.get_op_functions("custom_ops", "my_identity")
        self.assertEqual(len(functions), 1)
        self.assertIs(functions[0].onnx_function, custom_identity)
        self.assertTrue(functions[0].is_custom)


if __name__ == "__main__":
    run_tests()
