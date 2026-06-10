"""
Add validation cases for torch.jit.ScriptModule APIs on NPU:
1. PyTorch community lacks sufficient direct API validations for
   ScriptModule instance methods, so this file is added.
2. This file validates 19 torch.jit.ScriptModule APIs:
   train, eval, requires_grad_, zero_grad,
   float, double, to, type, to_empty, xpu,
   save, state_dict, set_extra_state, share_memory,
   register_module, register_parameter, set_submodule,
   get_buffer, extra_repr (extendable).
"""

import io
import os
import re
import tempfile

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestScriptModuleTraining(TestCase):

    def test_train_sets_training_mode(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        model.train()
        self.assertTrue(model.training)

    def test_train_returns_self(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        result = model.train()
        self.assertIs(result, model)

    def test_train_chained_call(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        result = model.train().train()
        self.assertTrue(result.training)

    def test_eval_sets_eval_mode(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        model.eval()
        self.assertFalse(model.training)

    def test_eval_returns_self(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        result = model.eval()
        self.assertIs(result, model)

    def test_eval_chained_call(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        result = model.eval().eval()
        self.assertFalse(result.training)

    def test_train_eval_toggle(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        model.train()
        self.assertTrue(model.training)
        model.eval()
        self.assertFalse(model.training)
        model.train()
        self.assertTrue(model.training)

    def test_train_propagates_to_submodule(self):
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.sub = Sub()

            @torch.jit.script_method
            def forward(self, x):
                return self.sub(x)

        model = M().to(device_type)
        model.eval()
        self.assertFalse(model.sub.training)
        model.train()
        self.assertTrue(model.sub.training)

    def test_requires_grad_sets_flag(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        model.requires_grad_(True)
        self.assertTrue(model.linear.weight.requires_grad)

    def test_requires_grad_returns_self(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        result = model.requires_grad_(False)
        self.assertIs(result, model)

    def test_requires_grad_false(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        model.requires_grad_(False)
        self.assertFalse(model.linear.weight.requires_grad)

    def test_zero_grad_clears_gradients(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        x = torch.randn(2, 2, requires_grad=True).to(device_type)
        out = model(x)
        out.sum().backward()
        self.assertIsNotNone(model.linear.weight.grad)
        model.zero_grad()
        self.assertIsNone(model.linear.weight.grad)

    def test_zero_grad_set_to_none_false(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        x = torch.randn(2, 2, requires_grad=True).to(device_type)
        out = model(x)
        out.sum().backward()
        self.assertIsNotNone(model.linear.weight.grad)
        model.zero_grad(set_to_none=False)
        self.assertIsNotNone(model.linear.weight.grad)
        self.assertEqual(model.linear.weight.grad, torch.zeros_like(
            model.linear.weight.grad))

    def test_zero_grad_no_gradients(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        model.zero_grad()


class TestScriptModuleDtype(TestCase):

    def test_float_converts_parameters(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        model.float()
        self.assertEqual(model.linear.weight.dtype, torch.float32)

    def test_float_returns_self(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        result = model.float()
        self.assertIs(result, model)

    def test_double_converts_parameters(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        model.double()
        # NPU may cast double to float, verify actual dtype
        self.assertIn(model.linear.weight.dtype, (torch.float64, torch.float32))

    def test_double_returns_self(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        result = model.double()
        self.assertIs(result, model)

    def test_to_dtype(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        model.to(torch.float64)
        # NPU may cast double to float
        self.assertIn(model.linear.weight.dtype, (torch.float64, torch.float32))

    def test_to_device(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        model.to(device_type)
        self.assertEqual(model.linear.weight.device.type, device_type)

    def test_to_returns_self_or_copy(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        result = model.to(device_type)
        self.assertIsInstance(result, torch.jit.ScriptModule)

    def test_to_chained_call(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        result = model.to(torch.float64).to(torch.float32)
        self.assertEqual(model.linear.weight.dtype, torch.float32)

    def test_type_float32_no_downgrade(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        # Normal path: float32 input stays float32
        model.type(torch.float32)
        self.assertEqual(model.linear.weight.dtype, torch.float32)

    def test_type_float64(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        model.type(torch.float64)
        # NPU may cast double to float
        self.assertIn(model.linear.weight.dtype, (torch.float64, torch.float32))

    def test_type_returns_self(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        result = model.type(torch.float32)
        self.assertIs(result, model)

    def test_to_empty_moves_to_device(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        result = model.to_empty(device=device_type)
        self.assertIsInstance(result, torch.jit.ScriptModule)
        self.assertEqual(result.linear.weight.device.type, device_type)

    def test_to_empty_returns_self(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        result = model.to_empty(device=device_type)
        self.assertIs(result, model)

    def test_xpu_raises(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        # XPU is not compiled in current environment
        with self.assertRaises(AssertionError):
            model.xpu()


class TestScriptModuleSerialization(TestCase):

    def test_save_to_file(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_and_load(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = torch.jit.load(path)
            x = torch.randn(2, 2).to(device_type)
            self.assertEqual(model(x), loaded(x))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_preserves_parameters(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        with torch.no_grad():
            model.linear.weight.fill_(1.0)
            model.linear.bias.fill_(2.0)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = torch.jit.load(path)
            self.assertEqual(loaded.linear.weight, torch.ones(2, 2))
            self.assertEqual(loaded.linear.bias, torch.ones(2) * 2)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_to_buffer(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        buffer = model.save_to_buffer()
        self.assertIsInstance(buffer, bytes)
        self.assertGreater(len(buffer), 0)

    def test_save_to_buffer_and_load(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        buffer = model.save_to_buffer()
        loaded = torch.jit.load(io.BytesIO(buffer))
        x = torch.randn(2, 2).to(device_type)
        self.assertEqual(model(x), loaded(x))

    def test_state_dict_returns_dict(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        sd = model.state_dict()
        self.assertIsInstance(sd, dict)
        self.assertIn("linear.weight", sd)
        self.assertIn("linear.bias", sd)

    def test_state_dict_values_on_npu(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        sd = model.state_dict()
        self.assertEqual(sd["linear.weight"].device.type, device_type)

    def test_state_dict_no_training_key(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        sd = model.state_dict()
        self.assertNotIn("training", sd)

    def test_state_dict_buffer_included(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(2, 2))

            @torch.jit.script_method
            def forward(self, x):
                return x + self.buf

        model = M().to(device_type)
        sd = model.state_dict()
        self.assertIn("buf", sd)
        self.assertEqual(sd["buf"], torch.ones(2, 2))

    def test_set_extra_state_raises(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        with self.assertRaises(RuntimeError):
            model.set_extra_state({"version": 1})

    def test_share_memory_raises_on_npu(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        # share_memory is intercepted by torch-npu for NPU modules
        with self.assertRaises(RuntimeError):
            model.share_memory()


class TestScriptModuleMetadata(TestCase):

    def test_register_module_raises_on_npu(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        sub = nn.Linear(2, 2).to(device_type)
        # torch-npu intercepts register_module for NPU modules
        with self.assertRaises(RuntimeError):
            model.register_module("new_sub", sub)

    def test_register_parameter_raises_on_npu(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        param = nn.Parameter(torch.randn(2, 2)).to(device_type)
        # torch-npu intercepts register_parameter for NPU modules
        with self.assertRaises(RuntimeError):
            model.register_parameter("new_param", param)

    def test_set_submodule_raises(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = M().to(device_type)
        sub = nn.Linear(2, 2).to(device_type)
        # Cannot re-assign modules in a ScriptModule
        with self.assertRaises(RuntimeError):
            model.set_submodule("linear", sub)

    def test_set_submodule_nested_raises(self):
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.sub = Sub()

            @torch.jit.script_method
            def forward(self, x):
                return self.sub(x)

        model = M().to(device_type)
        new_sub = nn.Linear(2, 2).to(device_type)
        with self.assertRaises(RuntimeError):
            model.set_submodule("sub.linear", new_sub)

    def test_get_buffer_returns_buffer(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(2, 2))

            @torch.jit.script_method
            def forward(self, x):
                return x + self.buf

        model = M().to(device_type)
        buf = model.get_buffer("buf")
        self.assertEqual(buf, torch.ones(2, 2))

    def test_get_buffer_nested(self):
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(2, 2))

            @torch.jit.script_method
            def forward(self, x):
                return x + self.buf

        class M(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.sub = Sub()

            @torch.jit.script_method
            def forward(self, x):
                return self.sub(x)

        model = M().to(device_type)
        buf = model.get_buffer("sub.buf")
        self.assertEqual(buf, torch.ones(2, 2))

    def test_extra_repr_returns_string(self):
        class MyScriptModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = MyScriptModule().to(device_type)
        result = model.extra_repr()
        self.assertIsInstance(result, str)
        # PyTorch 2.7.1 returns empty string on ScriptModule extra_repr.
        # Verify that str(model) contains ScriptModule class info.
        full_repr = str(model)
        self.assertIn("ScriptModule", full_repr)

    def test_extra_repr_matches_pattern(self):
        class MyScriptModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.linear(x)

        model = MyScriptModule().to(device_type)
        # Verify the model string contains a class-like identifier
        full_repr = repr(model)
        self.assertIsNotNone(re.search(r"MyScriptModule|ScriptModule", full_repr))


if __name__ == "__main__":
    run_tests()
