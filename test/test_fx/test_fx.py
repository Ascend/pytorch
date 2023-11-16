# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
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

import unittest

import functools
from operator import mod
from typing import Any, Callable, Dict, NamedTuple, List, Optional, Tuple, Union
import unittest

import torch
from torch.fx import Proxy, Node, GraphModule, Interpreter, Tracer, Graph, wrap
from torch.fx.node import Target, Argument
from torch.fx.proxy import TraceError

from torch_npu.fx import symbolic_trace
from torch_npu.testing.testcase import TestCase, run_tests
import torch_npu

try:
    from torchvision.models import resnet18
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


def a_non_torch_leaf(a, b):
    return a + b

# Test wrap() passing both a function name as well as a function
# directly


def a_lifted_leaf(a, b):
    return a[0] + a[1] + b


wrap('a_lifted_leaf')


def a_lifted_leaf2(a, b):
    return a[0] + a[1] + b


wrap(a_lifted_leaf2)

real_a_lifed_leaf = a_lifted_leaf
real_a_lifed_leaf2 = a_lifted_leaf2


class TestFX(TestCase):
    def checkGraphModule(self, m: torch.nn.Module, args, kwargs=None):
        """Check that an nn.Module's results match the GraphModule version
        for a given set of args/kwargs.
        """
        kwargs = kwargs or {}
        ref_outs = m(*args, **kwargs)
        gm = symbolic_trace(m)
        gm.graph.lint()
        test_outs = gm(*args, **kwargs)
        self.assertEqual(ref_outs, test_outs)

    def test_graph_module(self):
        class MySub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch_npu.rand(4, 3))

            def forward(self, x):
                return self.w + x

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.sub_mod = MySub()
                self.w = torch.nn.Parameter(torch.rand(3).npu())

            def forward(self, A, B, c):
                t = torch.sigmoid(A) + self.lin(c)
                return self.sub_mod(t.data + self.w + t + 1 - A + B // A + -A + A.add(B, alpha=3))

        m = MyModule()
        gm = symbolic_trace(m)

        class M2(torch.nn.Module):
            def forward(self, A):
                m, idx = torch.max(A, 0)
                return m + 1, idx + 1

        m2 = M2()
        gm2 = symbolic_trace(m2)

        class T(torch.nn.Module):
            def forward(self, A, b=4, *args, c=5, **kwargs):
                x = A + 1 + args.get(0) + kwargs.get('3')
                return x

        t = T()
        symbolic_trace(t)

    def test_custom_import(self):
        graph = torch.fx.Graph()
        a = graph.placeholder('x')
        b = graph.placeholder('y')
        c = graph.call_function(a_non_torch_leaf, (a, b))
        d = graph.call_function(torch.sin, (c,))
        graph.output(d)
        gm = GraphModule(torch.nn.Module(), graph)
        x, y = torch.rand(1).npu(), torch.rand(1).npu()
        self.assertEqual(torch.sin(x + y), gm(x, y))

    def test_args_kwargs(self):
        class T(torch.nn.Module):
            def forward(self, *args, **kwargs):
                x = args[0] + kwargs.get('foo')
                return x

        t = T()
        self.checkGraphModule(t, (torch.rand(1).npu(), torch.rand(1).npu()), {'foo': torch.rand(1).npu()})

    def test_args_kwargs_no_self(self):
        class T(torch.nn.Module):
            def forward(*args, **kwargs):  # noqa: B902
                self = args[0]
                return torch.relu(args[1])

        t = T()
        with self.assertRaisesRegex(RuntimeError, r'cannot be part of \*args expansion'):
            self.checkGraphModule(t, (torch.rand(1).npu(), torch.rand(1).npu()), {'foo': torch.rand(1).npu()})

    def test_fx_shifts(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x << 3, x >> 3

        input_tensor = torch.LongTensor(10).random_(0, 1024).npu()

        m = MyModule()
        self.checkGraphModule(m, (input_tensor,))

    def test_dict(self):
        class MyDictMod(torch.nn.Module):
            def forward(self, d):
                return d['3'].relu(), {'4': d['3'].neg()}

        input_dict = {'3': torch.rand(3, 4).npu()}
        m = MyDictMod()

        self.checkGraphModule(m, (input_dict,))

    @skipIfNoTorchVision
    def test_resnet(self):
        resnet = resnet18().to('npu')
        resnet.eval()

        res_graph = symbolic_trace(resnet)

        ip = torch.rand(1, 3, 224, 224).to('npu')
        a = resnet(ip)
        b = res_graph(ip)

        self.assertEqual(a, b)

    def test_unpack(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                c, d = a
                return c + d + b

        a = (torch.rand(1).npu(), torch.rand(1).npu())
        b = torch.rand(1).npu()
        m = M()
        self.checkGraphModule(m, (a, b))

    def test_tensor_attribute(self):
        class TensorAttribute(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tensor = torch.rand(3, 4).npu()

            def forward(self, x):
                return torch.nn.functional.linear(x, self.tensor)

        ta = TensorAttribute()
        traced = symbolic_trace(ta)
        traced(torch.rand(4, 4).npu())

        class WrapperForQualname(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ta = TensorAttribute()

            def forward(self, x):
                return torch.nn.functional.linear(x, self.ta.tensor)

        wfq = WrapperForQualname()
        traced2 = symbolic_trace(wfq)
        traced2.graph.lint()
        traced2(torch.rand(4, 4).npu())

    def test_symbolic_trace_sequential(self):
        class Simple(torch.nn.Module):
            def forward(self, x):
                return torch.neg(x)

        seq = torch.nn.Sequential(
            Simple(),
            Simple(),
            Simple()
        )
        traced = symbolic_trace(seq)
        traced.graph.lint()
        x = torch.rand(3, 4).npu()
        self.assertEqual(traced(x), seq(x))

    def test_tensor_constant(self):
        class ConstTensor(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.linear(x, torch.zeros(3, 4).npu())

        ct = ConstTensor()
        traced = symbolic_trace(ct)
        traced.graph.lint()
        traced(torch.rand(4, 4).npu())

    def test_unpack_list_better_error(self):
        class SomeArgs(torch.nn.Module):
            def forward(self, a, b):
                return torch.rand(3, 4).npu()

        class UnpacksList(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sa = SomeArgs()

            def forward(self, x: list):
                return self.sa(*x)

        ul = UnpacksList()
        with self.assertRaisesRegex(TraceError, 'Proxy object cannot be iterated.'):
            symbolic_trace(ul)

    def test_unpack_dict_better_error(self):
        class SomeKwargs(torch.nn.Module):
            def forward(self, x=3, y=4):
                return torch.rand(3, 4).npu()

        class UnpacksDict(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sk = SomeKwargs()

            def forward(self, x: dict):
                return self.sk(**x)

        ud = UnpacksDict()
        with self.assertRaisesRegex(TraceError, 'Proxy object cannot be iterated.'):
            symbolic_trace(ud)

    @unittest.skip("skip test_npu_contrib_function_trace now")
    def test_npu_contrib_function_trace(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch_npu.contrib.function.npu_diou(x, x)

        module = MyModule()
        traced = symbolic_trace(module)
        traced.graph.lint()
        x = torch.rand(4, 3).npu()
        self.assertEqual(traced(x), module(x))

    def test_npu_contrib_module_trace(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mish = torch_npu.contrib.module.Mish()

            def forward(self, x):
                return self.mish(x)

        module = MyModule()
        traced = symbolic_trace(module)
        traced.graph.lint()
        x = torch.rand(4, 3).npu()
        self.assertEqual(traced(x), module(x))

    @unittest.skip("skip test_npu_custom_op_trace now")
    def test_npu_custom_op_trace(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.npu_format_cast(x, 2)

        module = MyModule()
        traced = symbolic_trace(module)
        traced.graph.lint()
        x = torch.rand(4, 3).npu()
        self.assertEqual(traced(x), module(x))


if __name__ == '__main__':
    run_tests()
