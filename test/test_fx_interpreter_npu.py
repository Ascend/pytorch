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
Add validation cases for torch.fx.Interpreter APIs on NPU:
1. PyTorch community lacks dedicated direct test cases for
   Interpreter.boxed_run, Interpreter.fetch_attr,
   Interpreter.map_nodes_to_values and
   Interpreter.fetch_args_kwargs_from_env, so this file is added.
2. This file validates these internal methods on NPU.
"""

import torch

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.fx import Interpreter, symbolic_trace


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestInterpreterBoxedRun(TestCase):
    """Test Interpreter.boxed_run method."""

    def test_boxed_run_basic(self):
        class AddModule(torch.nn.Module):
            def forward(self, lhs, rhs):
                return lhs + rhs

        gm = symbolic_trace(AddModule())
        interpreter = Interpreter(gm)
        lhs = torch.tensor(1.0, device=device_type)
        rhs = torch.tensor(2.0, device=device_type)
        result = interpreter.boxed_run([lhs.clone(), rhs.clone()])
        self.assertTrue(torch.equal(result, lhs + rhs))

    def test_boxed_run_clears_args(self):
        class AddModule(torch.nn.Module):
            def forward(self, lhs, rhs):
                return lhs + rhs

        gm = symbolic_trace(AddModule())
        interpreter = Interpreter(gm)
        lhs = torch.tensor(1.0, device=device_type)
        rhs = torch.tensor(2.0, device=device_type)
        args_list = [lhs.clone(), rhs.clone()]
        interpreter.boxed_run(args_list)
        self.assertEqual(args_list, [])


class TestInterpreterFetchAttr(TestCase):
    """Test Interpreter.fetch_attr method."""

    def test_fetch_attr_parameter(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(
                    torch.ones(2, 2, device=device_type))

            def forward(self, x):
                return x + self.param

        m = M()
        gm = symbolic_trace(m)
        interp = Interpreter(gm)
        param = interp.fetch_attr("param")
        self.assertTrue(torch.equal(param, torch.ones(2, 2, device=device_type)))

    def test_fetch_attr_submodule(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = torch.nn.Linear(3, 3).to(device_type)

            def forward(self, x):
                return self.sub(x)

        m = M()
        gm = symbolic_trace(m)
        interp = Interpreter(gm)
        sub = interp.fetch_attr("sub")
        self.assertIsInstance(sub, torch.nn.Module)


class TestInterpreterMapNodesToValues(TestCase):
    """Test Interpreter.map_nodes_to_values method."""

    def test_map_nodes_to_values_args(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        gm = symbolic_trace(M())
        interp = Interpreter(gm)
        x = torch.ones(2, 2, device=device_type)
        y = torch.zeros(2, 2, device=device_type)
        interp.args_iter = iter([x, y])
        add_node = [n for n in gm.graph.nodes if n.op == "call_function"][0]
        # fill env first so map_nodes_to_values can replace Node with values
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        for n in placeholder_nodes:
            interp.env[n] = next(interp.args_iter)
        mapped = interp.map_nodes_to_values(add_node.args, add_node)
        self.assertIsInstance(mapped, tuple)
        self.assertTrue(torch.equal(mapped[0], x))
        self.assertTrue(torch.equal(mapped[1], y))

    def test_map_nodes_to_values_kwargs(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, other=1)

        gm = symbolic_trace(M())
        interp = Interpreter(gm)
        x = torch.ones(2, 2, device=device_type)
        interp.args_iter = iter([x])
        add_node = [n for n in gm.graph.nodes if n.op == "call_function"][0]
        self.assertEqual(add_node.kwargs, {"other": 1})
        mapped = interp.map_nodes_to_values(add_node.kwargs, add_node)
        self.assertIsInstance(mapped, dict)
        self.assertEqual(mapped, {"other": 1})


class TestInterpreterFetchArgsKwargsFromEnv(TestCase):
    """Test Interpreter.fetch_args_kwargs_from_env method."""

    def test_fetch_args_kwargs_from_env(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.add(x, y)

        gm = symbolic_trace(M())
        interp = Interpreter(gm)
        x = torch.ones(2, 2, device=device_type)
        y = torch.zeros(2, 2, device=device_type)
        interp.args_iter = iter([x, y])
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        add_node = [n for n in gm.graph.nodes if n.op == "call_function"][0]
        interp.env = {}
        for n in placeholder_nodes:
            interp.env[n] = next(interp.args_iter)
        args, kwargs = interp.fetch_args_kwargs_from_env(add_node)
        self.assertEqual(len(args), 2)
        self.assertIsInstance(kwargs, dict)
        self.assertTrue(torch.equal(args[0], x))
        self.assertTrue(torch.equal(args[1], y))

    def test_fetch_args_kwargs_from_env_non_empty_kwargs(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, other=1)

        gm = symbolic_trace(M())
        interp = Interpreter(gm)
        x = torch.ones(2, 2, device=device_type)
        interp.args_iter = iter([x])
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        add_node = [n for n in gm.graph.nodes if n.op == "call_function"][0]
        interp.env = {}
        for n in placeholder_nodes:
            interp.env[n] = next(interp.args_iter)
        args, kwargs = interp.fetch_args_kwargs_from_env(add_node)
        self.assertEqual(args, (x,))
        self.assertEqual(kwargs, {"other": 1})


if __name__ == "__main__":
    run_tests()
