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
Add validation cases for torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts on NPU.

1. PyTorch community lacks sufficient and direct API validations for PropagateUnbackedSymInts,
   so this file is added.
2. This file validates PropagateUnbackedSymInts.run, PropagateUnbackedSymInts.run_node,
   PropagateUnbackedSymInts.placeholder, PropagateUnbackedSymInts.output,
   PropagateUnbackedSymInts.boxed_run, PropagateUnbackedSymInts.call_function,
   PropagateUnbackedSymInts.call_method, and rebind_unbacked.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._dynamo.utils import detect_fake_mode
from torch.fx import Interpreter, symbolic_trace
from torch.fx.experimental.symbolic_shapes import PropagateUnbackedSymInts
from torch_npu.utils._dynamo import _dynamo_register_interface_for_device

# Ensure NPU Dynamo device interface is registered before torch.export(strict=True).
# has_triton() may query "npu" before lazy inductor/init registration runs.
_dynamo_register_interface_for_device()

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
torch.zeros(3, 4).to(device_type)


class TestPropagateUnbackedSymInts(TestCase):

    def test_propagate_unbacked_symints_run(self):
        """Test PropagateUnbackedSymInts.run with NPU tensor."""

        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.nonzero(x)

        inp = (torch.tensor([1, 0, 1, 0]).to(device_type),)
        gm = torch.export.export(M(), inp, strict=True).module()
        fake_inputs = [
            node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"
        ]
        fake_mode = detect_fake_mode(fake_inputs)
        with fake_mode:
            result = PropagateUnbackedSymInts(gm).run(*fake_inputs)
        self.assertIsNotNone(result)

    def test_propagate_unbacked_symints_run_node(self):
        """Test PropagateUnbackedSymInts.run_node with NPU tensor."""

        class RunNodeCapturingInterpreter(PropagateUnbackedSymInts):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.captured_results = {}

            def run_node(self, n):
                result = super().run_node(n)
                self.captured_results[n] = result
                return result

        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.nonzero(x)

        inp = (torch.tensor([1, 0, 1, 0]).to(device_type),)
        gm = torch.export.export(M(), inp, strict=True).module()
        fake_inputs = [
            node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"
        ]
        fake_mode = detect_fake_mode(fake_inputs)
        with fake_mode:
            interpreter = RunNodeCapturingInterpreter(gm)
            result = interpreter.run(*fake_inputs)
            self.assertIsNotNone(result)
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    self.assertIn(node, interpreter.captured_results)

    def test_propagate_unbacked_symints_placeholder(self):
        """Test PropagateUnbackedSymInts.placeholder with NPU tensor."""

        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.nonzero(x)

        inp = (torch.tensor([1, 0, 1, 0]).to(device_type),)
        gm = torch.export.export(M(), inp, strict=True).module()
        fake_inputs = [
            node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"
        ]
        fake_mode = detect_fake_mode(fake_inputs)
        with fake_mode:
            interpreter = PropagateUnbackedSymInts(gm)
            interpreter.args_iter = iter(fake_inputs)
            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    result = interpreter.placeholder(node.target, node.args, node.kwargs)
                    self.assertIsNotNone(result)

    def test_propagate_unbacked_symints_output(self):
        """Test PropagateUnbackedSymInts.output with NPU tensor."""

        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.nonzero(x)

        inp = (torch.tensor([1, 0, 1, 0]).to(device_type),)
        gm = torch.export.export(M(), inp, strict=True).module()
        fake_inputs = [
            node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"
        ]
        fake_mode = detect_fake_mode(fake_inputs)
        with fake_mode:
            interpreter = PropagateUnbackedSymInts(gm)
            interpreter.run(*fake_inputs)
            for node in gm.graph.nodes:
                if node.op == "output":
                    result = interpreter.output(node.target, node.args, node.kwargs)
                    self.assertIsNotNone(result)

    def test_rebind_unbacked(self):
        """Test rebind_unbacked with NPU tensor."""

        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.nonzero(x)

        inp = (torch.tensor([1, 0, 1, 0]).to(device_type),)
        gm = torch.export.export(M(), inp, strict=True).module()
        fake_inputs = [
            node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"
        ]
        fake_mode = detect_fake_mode(fake_inputs)
        shape_prop_gm = torch.fx.passes.shape_prop.ShapeProp(
            gm=gm, fake_mode=fake_mode
        )
        shape_prop_gm.propagate(*fake_inputs)
        self.assertEqual(len(fake_mode.shape_env.pending_fresh_unbacked_symbols), 0)

    def test_propagate_unbacked_symints_boxed_run(self):
        """Test PropagateUnbackedSymInts.boxed_run with NPU tensor."""

        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.nonzero(x)

        inp = (torch.tensor([1, 0, 1, 0]).to(device_type),)
        gm = torch.export.export(M(), inp, strict=True).module()
        fake_inputs = [
            node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"
        ]
        fake_mode = detect_fake_mode(fake_inputs)
        with fake_mode:
            interpreter = PropagateUnbackedSymInts(gm)
            args_list = list(fake_inputs)
            result = interpreter.boxed_run(args_list)
        self.assertIsNotNone(result)

    def test_propagate_unbacked_symints_call_function(self):
        """Test PropagateUnbackedSymInts.call_function with NPU tensor."""
        self.assertIs(
            PropagateUnbackedSymInts.call_function,
            Interpreter.call_function,
        )

        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.add(x, x)

        gm = symbolic_trace(M())
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        call_function = next(
            node for node in gm.graph.nodes if node.op == "call_function"
        )

        interpreter = PropagateUnbackedSymInts(gm)
        interpreter.env[placeholder] = torch.ones(2, 3).to(device_type)

        args, kwargs = interpreter.fetch_args_kwargs_from_env(call_function)
        result = interpreter.call_function(call_function.target, args, kwargs)
        self.assertEqual(tuple(result.shape), (2, 3))
        self.assertEqual(result.device.type, device_type)

    def test_propagate_unbacked_symints_call_method(self):
        """Test PropagateUnbackedSymInts.call_method with NPU tensor."""
        self.assertIs(
            PropagateUnbackedSymInts.call_method,
            Interpreter.call_method,
        )

        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x.relu()

        gm = symbolic_trace(M())
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        call_method = next(node for node in gm.graph.nodes if node.op == "call_method")

        interpreter = PropagateUnbackedSymInts(gm)
        interpreter.env[placeholder] = torch.randn(2, 3).to(device_type)

        args, kwargs = interpreter.fetch_args_kwargs_from_env(call_method)
        result = interpreter.call_method(call_method.target, args, kwargs)
        self.assertEqual(tuple(result.shape), (2, 3))
        self.assertEqual(result.device.type, device_type)


if __name__ == "__main__":
    run_tests()
