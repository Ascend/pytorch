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
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.onnx.ONNXProgram.optimize, torch.onnx.ONNXProgram.model_proto (extendable).
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


class TestONNXProgramModelProto(common_utils.TestCase):

    def test_model_proto_returns_valid_proto(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        x = torch.randn(3).to(device_type)
        onnx_program = torch.onnx.export(Model(), (x,), dynamo=True)
        proto = onnx_program.model_proto
        self.assertIsNotNone(proto)
        self.assertGreater(proto.ir_version, 0)

    def test_model_proto_graph_structure(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x * 2

        x = torch.randn(3).to(device_type)
        onnx_program = torch.onnx.export(Model(), (x,), dynamo=True)
        proto = onnx_program.model_proto
        self.assertEqual(len(proto.graph.input), 1)
        self.assertEqual(len(proto.graph.output), 1)

    def test_model_proto_producer_name(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x

        x = torch.randn(3).to(device_type)
        onnx_program = torch.onnx.export(Model(), (x,), dynamo=True)
        proto = onnx_program.model_proto
        self.assertEqual(proto.producer_name, "pytorch")

    def test_model_proto_serialization(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + x

        x = torch.randn(3).to(device_type)
        onnx_program = torch.onnx.export(Model(), (x,), dynamo=True)
        proto = onnx_program.model_proto
        serialized = proto.SerializeToString()
        self.assertGreater(len(serialized), 0)

    def test_model_proto_multiple_inputs_outputs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y, x - y

        x = torch.randn(3).to(device_type)
        y = torch.randn(3).to(device_type)
        onnx_program = torch.onnx.export(Model(), (x, y), dynamo=True)
        proto = onnx_program.model_proto
        self.assertEqual(len(proto.graph.input), 2)
        self.assertEqual(len(proto.graph.output), 2)


if __name__ == "__main__":
    common_utils.run_tests()
