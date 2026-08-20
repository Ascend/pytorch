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
Add validation cases for torch.onnx.ONNXProgram APIs on NPU:
1. PyTorch community lacks direct API validations for ONNXProgram.optimize, so this file is added.
2. This file validates torch.onnx.ONNXProgram.optimize (extendable).
"""

from __future__ import annotations

import torch
from torch.testing._internal import common_utils

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class ONNXProgramOptimizeTest(common_utils.TestCase):
    """Tests for ONNXProgram.optimize method."""

    def _create_onnx_program(self) -> torch.onnx.ONNXProgram:
        """Helper to create an unoptimized ONNXProgram from a simple model."""
        class Model(torch.nn.Module):
            def forward(self, x):
                return (x + 1) * 2

        x = torch.randn(3).to(device_type)
        onnx_program = torch.onnx.export(
            Model().eval(), (x,), dynamo=True, optimize=False, verbose=False
        )
        return onnx_program

    def test_optimize_returns_none(self):
        """optimize() should return None."""
        onnx_program = self._create_onnx_program()
        result = onnx_program.optimize()
        self.assertIsNone(result)

    def test_optimize_model_valid(self):
        """The model should be valid after optimize()."""
        onnx_program = self._create_onnx_program()
        onnx_program.optimize()
        self.assertIsNotNone(onnx_program.model)
        self.assertIsNotNone(onnx_program.model.graph)

    def test_optimize_idempotent(self):
        """Calling optimize() twice should not raise."""
        onnx_program = self._create_onnx_program()
        onnx_program.optimize()
        onnx_program.optimize()

    def test_optimize_preserves_io_count(self):
        """The optimized model should preserve graph input/output count."""
        onnx_program = self._create_onnx_program()
        num_inputs = len(onnx_program.model.graph.inputs)
        num_outputs = len(onnx_program.model.graph.outputs)
        onnx_program.optimize()
        self.assertEqual(len(onnx_program.model.graph.inputs), num_inputs)
        self.assertEqual(len(onnx_program.model.graph.outputs), num_outputs)


if __name__ == "__main__":
    common_utils.run_tests()
