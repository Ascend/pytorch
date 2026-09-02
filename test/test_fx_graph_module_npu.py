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
Add validation cases for torch.fx.graph_module APIs on NPU:
1. PyTorch community lacks dedicated test cases for internal
   graph_module APIs, so this file is added.
2. This file validates _exec_with_source, _forward_from_src,
   _CodeOnlyModule, _copy_attr, and _WrappedCall.
"""

import torch

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.fx.graph_module import (
    _exec_with_source,
    _forward_from_src,
    _CodeOnlyModule,
    _copy_attr,
    _WrappedCall,
    _loader,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestExecWithSource(TestCase):
    """Test _exec_with_source function."""

    def test_exec_with_source_basic(self):
        src = "x = 42"
        g = {}
        _exec_with_source(src, g)
        self.assertEqual(g["x"], 42)

    def test_exec_with_source_multiple(self):
        src = "a = 1\nb = 2\nc = a + b"
        g = {}
        _exec_with_source(src, g)
        self.assertEqual(g["a"], 1)
        self.assertEqual(g["b"], 2)
        self.assertEqual(g["c"], 3)

    def test_exec_with_source_invalid_syntax(self):
        src = "x = "
        g = {}
        with self.assertRaises(SyntaxError):
            _exec_with_source(src, g)

    def test_exec_with_source_invalid_globals(self):
        src = "x = 42"
        with self.assertRaises((AttributeError, TypeError)):
            _exec_with_source(src, None)

    def test_exec_with_source_co_fields(self):
        src = "x = 42"
        g = {}
        co_fields = {"co_filename": "test_mod.py", "co_firstlineno": 10, "co_name": "test_exec"}
        _exec_with_source(src, g, co_fields)
        self.assertEqual(g["x"], 42)
        cache_keys = list(_loader.eval_cache.keys())
        self.assertTrue(any("test_mod.py:10 in test_exec" in k for k in cache_keys))


class TestForwardFromSrc(TestCase):
    """Test _forward_from_src function."""

    def test_forward_from_src_basic(self):
        src = (
            "import torch\n"
            "def forward(self, x):\n"
            "    return x + 1\n"
        )
        fn = _forward_from_src(src, {})
        x = torch.tensor(1.0).to(device_type)
        result = fn(None, x)
        self.assertEqual(result, torch.tensor(2.0).to(device_type))

    def test_forward_from_src_with_imports(self):
        src = (
            "import torch\n"
            "def forward(self, x):\n"
            "    return torch.relu(x)\n"
        )
        fn = _forward_from_src(src, {})
        t = torch.tensor([-1.0, 0.0, 1.0]).to(device_type)
        result = fn(None, t)
        expected = torch.tensor([0.0, 0.0, 1.0]).to(device_type)
        self.assertEqual(result, expected)

    def test_forward_from_src_missing_forward(self):
        src = "x = 1"
        with self.assertRaises((KeyError, SyntaxError)):
            _forward_from_src(src, {})

    def test_forward_from_src_co_fields(self):
        src = (
            "def forward(self, x):\n"
            "    return x + 1\n"
        )
        co_fields = {"co_filename": "fwd_mod.py", "co_firstlineno": 5, "co_name": "forward_src"}
        fn = _forward_from_src(src, {}, co_fields)
        x = torch.tensor(1.0).to(device_type)
        result = fn(None, x)
        self.assertEqual(result, torch.tensor(2.0).to(device_type))
        cache_keys = list(_loader.eval_cache.keys())
        self.assertTrue(any("fwd_mod.py:5 in forward_src" in k for k in cache_keys))


class TestCodeOnlyModule(TestCase):
    """Test _CodeOnlyModule class."""

    def test_code_only_module_basic(self):
        body = {"a": 1, "b": "test"}
        m = _CodeOnlyModule(body)
        self.assertEqual(m.a, 1)
        self.assertEqual(m.b, "test")

    def test_code_only_module_empty(self):
        m = _CodeOnlyModule({})
        self.assertIsInstance(m, torch.nn.Module)


class TestCopyAttr(TestCase):
    """Test _copy_attr function."""

    def test_copy_attr_tensor(self):
        src_mod = torch.nn.Module()
        dst_mod = torch.nn.Module()
        src_mod.register_buffer("weight", torch.ones(3, 4).to(device_type))
        _copy_attr(src_mod, dst_mod, "weight")
        self.assertTrue(hasattr(dst_mod, "weight"))
        self.assertEqual(dst_mod.weight, torch.ones(3, 4).to(device_type))

    def test_copy_attr_parameter(self):
        src_mod = torch.nn.Module()
        dst_mod = torch.nn.Module()
        src_mod.register_parameter(
            "param", torch.nn.Parameter(torch.zeros(2, 2).to(device_type)))
        _copy_attr(src_mod, dst_mod, "param")
        self.assertTrue(hasattr(dst_mod, "param"))
        self.assertEqual(dst_mod.param, torch.zeros(2, 2).to(device_type))

    def test_copy_attr_nested(self):
        src_mod = torch.nn.Module()
        dst_mod = torch.nn.Module()
        child = torch.nn.Module()
        child.register_buffer("buf", torch.ones(2).to(device_type))
        src_mod.add_module("child", child)
        _copy_attr(src_mod, dst_mod, "child.buf")
        self.assertTrue(hasattr(dst_mod.child, "buf"))
        self.assertEqual(dst_mod.child.buf, torch.ones(2).to(device_type))

    def test_copy_attr_npu_tensor(self):
        src_mod = torch.nn.Module()
        dst_mod = torch.nn.Module()
        src_mod.register_buffer("npu_buf", torch.ones(3).to(device_type))
        _copy_attr(src_mod, dst_mod, "npu_buf")
        self.assertTrue(hasattr(dst_mod, "npu_buf"))
        self.assertEqual(dst_mod.npu_buf.device.type, device_type)

    def test_copy_attr_missing_attribute(self):
        src_mod = torch.nn.Module()
        dst_mod = torch.nn.Module()
        with self.assertRaises(AttributeError):
            _copy_attr(src_mod, dst_mod, "nonexistent")

    def test_copy_attr_existing_parent(self):
        src_mod = torch.nn.Module()
        child = torch.nn.Module()
        child.register_buffer("buf", torch.ones(2).to(device_type))
        src_mod.add_module("child", child)
        dst_mod = torch.nn.Module()
        dst_mod.add_module("child", torch.nn.Module())
        _copy_attr(src_mod, dst_mod, "child.buf")
        self.assertTrue(hasattr(dst_mod.child, "buf"))
        self.assertEqual(dst_mod.child.buf, torch.ones(2).to(device_type))


class TestWrappedCall(TestCase):
    """Test _WrappedCall class."""

    def test_wrapped_call_basic(self):
        class SimpleMod(torch.nn.Module):
            def forward(self, x):
                return x * 2

        mod = SimpleMod()
        wrapped = _WrappedCall(SimpleMod, None)
        t = torch.tensor(3.0).to(device_type)
        result = wrapped(mod, t)
        self.assertEqual(result, torch.tensor(6.0).to(device_type))

    def test_wrapped_call_with_cls_call(self):
        class SimpleMod(torch.nn.Module):
            def forward(self, x):
                return x + 1

        mod = SimpleMod()
        wrapped = _WrappedCall(SimpleMod, SimpleMod.forward)
        t = torch.tensor(5.0).to(device_type)
        result = wrapped(mod, t)
        self.assertEqual(result, torch.tensor(6.0).to(device_type))

    def test_wrapped_call_error_path(self):
        class BadMod(torch.nn.Module):
            def forward(self, x):
                return x.undefined_attr

        mod = BadMod()
        wrapped = _WrappedCall(BadMod, None)
        t = torch.tensor(1.0).to(device_type)
        with self.assertRaises(AttributeError):
            wrapped(mod, t)


if __name__ == "__main__":
    run_tests()
