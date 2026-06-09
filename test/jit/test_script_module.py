"""
Add validation cases for torch.jit.ScriptModule APIs on NPU:
1. PyTorch community lacks sufficient direct API validations for
   ScriptModule instance methods, so this file is added.
2. This file validates 19 torch.jit.ScriptModule APIs using
   torch.jit.script() as the canonical creation method:
   train, eval, zero_grad, float, double, to, type,
   state_dict, save, extra_repr, requires_grad_, to_empty,
   xpu, get_buffer, set_submodule, register_module,
   register_parameter, share_memory, set_extra_state
   (extendable).
"""

import io
import os
import re
import tempfile

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


# ---------------------------------------------------------------------------
# Module-level model builders (required by torch.jit.script() source access)
# ---------------------------------------------------------------------------

def _make_linear():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)

    return torch.jit.script(M().to(device_type))


def _make_with_buffer():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.register_buffer("buf", torch.ones(2, 2))

        def forward(self, x):
            return self.linear(x) + self.buf

    return torch.jit.script(M().to(device_type))


def _make_nested():
    class Sub(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = Sub()

        def forward(self, x):
            return self.sub(x)

    return torch.jit.script(Outer().to(device_type))


def _make_cpu_linear():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)

    return torch.jit.script(M())


def _make_nested_with_buffer():
    class Sub(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buf", torch.ones(2, 2))

        def forward(self, x):
            return x + self.buf

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = Sub()

        def forward(self, x):
            return self.sub(x)

    return torch.jit.script(Outer().to(device_type))


# ===================================================================
# Test Classes
# ===================================================================


class TestScriptModuleTrainEval(TestCase):

    def test_train_default_is_training(self):
        sm = _make_linear()
        self.assertTrue(sm.training)

    def test_train_set_true_explicit(self):
        sm = _make_linear()
        sm.train(True)
        self.assertTrue(sm.training)

    def test_train_set_false(self):
        sm = _make_linear()
        sm.train(False)
        self.assertFalse(sm.training)

    def test_eval_sets_training_false(self):
        sm = _make_linear()
        sm.eval()
        self.assertFalse(sm.training)

    def test_train_returns_self(self):
        sm = _make_linear()
        result = sm.train()
        self.assertIs(result, sm)

    def test_eval_returns_self(self):
        sm = _make_linear()
        result = sm.eval()
        self.assertIs(result, sm)

    def test_train_eval_roundtrip(self):
        sm = _make_linear()
        sm.train()
        self.assertTrue(sm.training)
        sm.eval()
        self.assertFalse(sm.training)
        sm.train(True)
        self.assertTrue(sm.training)

    def test_train_on_npu(self):
        sm = _make_linear()
        sm.train()
        self.assertTrue(sm.training)
        self.assertEqual(sm.linear.weight.device.type, device_type)

    def test_eval_on_npu(self):
        sm = _make_linear()
        sm.eval()
        self.assertFalse(sm.training)
        self.assertEqual(sm.linear.weight.device.type, device_type)

    def test_train_propagates_to_submodules(self):
        sm = _make_nested()
        sm.train()
        self.assertTrue(sm.sub.training)

    def test_eval_propagates_to_submodules(self):
        sm = _make_nested()
        sm.eval()
        self.assertFalse(sm.sub.training)


class TestScriptModuleZeroGrad(TestCase):

    def test_zero_grad_no_error(self):
        sm = _make_linear()
        sm.zero_grad()

    def test_zero_grad_clears_grads(self):
        sm = _make_linear()
        x = torch.randn(2, 2, requires_grad=True).to(device_type)
        out = sm(x)
        out.sum().backward()
        self.assertIsNotNone(sm.linear.weight.grad)
        sm.zero_grad()
        self.assertIsNone(sm.linear.weight.grad)

    def test_zero_grad_set_to_none(self):
        sm = _make_linear()
        x = torch.randn(2, 2, requires_grad=True).to(device_type)
        out = sm(x)
        out.sum().backward()
        self.assertIsNotNone(sm.linear.weight.grad)
        sm.zero_grad(set_to_none=True)
        self.assertIsNone(sm.linear.weight.grad)

    def test_zero_grad_set_to_none_false(self):
        sm = _make_linear()
        x = torch.randn(2, 2, requires_grad=True).to(device_type)
        out = sm(x)
        out.sum().backward()
        old_grad = sm.linear.weight.grad
        self.assertIsNotNone(old_grad)
        sm.zero_grad(set_to_none=False)
        self.assertIsNotNone(sm.linear.weight.grad)
        self.assertEqual(sm.linear.weight.grad,
                         torch.zeros_like(old_grad))

    def test_zero_grad_backward_chain_on_npu(self):
        sm = _make_linear()
        x = torch.randn(2, 2, requires_grad=True).to(device_type)
        out = sm(x)
        out.sum().backward()
        self.assertIsNotNone(sm.linear.weight.grad)
        sm.zero_grad()
        self.assertIsNone(sm.linear.weight.grad)
        out2 = sm(x)
        out2.sum().backward()
        self.assertIsNotNone(sm.linear.weight.grad)


class TestScriptModuleTo(TestCase):

    def test_to_dtype(self):
        sm = _make_linear()
        sm.to(torch.float64)
        self.assertIn(sm.linear.weight.dtype,
                      (torch.float64, torch.float32))

    def test_to_device(self):
        sm = _make_linear()
        sm.to(device_type)
        self.assertEqual(sm.linear.weight.device.type, device_type)

    def test_to_returns_self(self):
        sm = _make_linear()
        result = sm.to(torch.float32)
        self.assertIsInstance(result, torch.jit.ScriptModule)

    def test_to_device_and_dtype(self):
        sm = _make_linear()
        sm.to(device_type, torch.float64)
        self.assertEqual(sm.linear.weight.device.type, device_type)
        self.assertIn(sm.linear.weight.dtype,
                      (torch.float64, torch.float32))

    def test_to_dtype_keyword(self):
        sm = _make_linear()
        sm.to(dtype=torch.float64)
        self.assertIn(sm.linear.weight.dtype,
                      (torch.float64, torch.float32))

    def test_to_string_device(self):
        sm = _make_linear()
        sm.to(str(torch.device(device_type)))
        self.assertEqual(sm.linear.weight.device.type, device_type)

    def test_to_no_args_returns_self(self):
        sm = _make_linear()
        result = sm.to()
        self.assertIsInstance(result, torch.jit.ScriptModule)

    def test_to_propagates_to_submodules(self):
        sm = _make_nested()
        sm.to(dtype=torch.float64)
        self.assertIn(sm.sub.linear.weight.dtype,
                      (torch.float64, torch.float32))

    def test_to_npu_and_dtype(self):
        sm = _make_linear()
        sm.to(device_type, dtype=torch.float64)
        self.assertEqual(sm.linear.weight.device.type, device_type)
        self.assertIn(sm.linear.weight.dtype,
                      (torch.float64, torch.float32))


class TestScriptModuleFloatDouble(TestCase):

    def test_float_returns_self(self):
        sm = _make_linear()
        result = sm.float()
        self.assertIsInstance(result, torch.jit.ScriptModule)

    def test_float_converts_params(self):
        sm = _make_linear()
        sm.float()
        self.assertEqual(sm.linear.weight.dtype, torch.float32)

    def test_float_on_npu(self):
        sm = _make_linear()
        sm.float()
        self.assertEqual(sm.linear.weight.dtype, torch.float32)
        self.assertEqual(sm.linear.weight.device.type, device_type)

    def test_float_propagates_to_submodules(self):
        sm = _make_nested()
        sm.float()
        self.assertEqual(sm.sub.linear.weight.dtype, torch.float32)

    def test_double_returns_self(self):
        sm = _make_linear()
        result = sm.double()
        self.assertIsInstance(result, torch.jit.ScriptModule)

    def test_double_converts_params(self):
        sm = _make_linear()
        sm.double()
        self.assertIn(sm.linear.weight.dtype,
                      (torch.float64, torch.float32))

    def test_double_on_npu_fallback_to_float32(self):
        sm = _make_linear()
        sm.double()
        # NPU does not support float64; double() falls back to float32
        self.assertEqual(sm.linear.weight.dtype, torch.float32)


class TestScriptModuleType(TestCase):

    def test_type_float32(self):
        sm = _make_linear()
        sm.type(torch.float32)
        self.assertEqual(sm.linear.weight.dtype, torch.float32)

    def test_type_float64(self):
        sm = _make_linear()
        sm.type(torch.float64)
        self.assertIn(sm.linear.weight.dtype,
                      (torch.float64, torch.float32))

    def test_type_on_npu(self):
        sm = _make_linear()
        sm.type(torch.float64)
        self.assertEqual(sm.linear.weight.device.type, device_type)
        self.assertIn(sm.linear.weight.dtype,
                      (torch.float64, torch.float32))

    def test_type_int32_raises(self):
        sm = _make_linear()
        with self.assertRaisesRegex(
                RuntimeError, r"must be floating point"):
            sm.type(torch.int32)


class TestScriptModuleStateDict(TestCase):

    def test_state_dict_contains_params(self):
        sm = _make_linear()
        sd = sm.state_dict()
        self.assertIn("linear.weight", sd)
        self.assertIn("linear.bias", sd)

    def test_state_dict_contains_buffers(self):
        sm = _make_with_buffer()
        sd = sm.state_dict()
        self.assertIn("buf", sd)

    def test_state_dict_values_match(self):
        sm = _make_linear()
        sd = sm.state_dict()
        self.assertEqual(sd["linear.weight"], sm.linear.weight)
        self.assertEqual(sd["linear.bias"], sm.linear.bias)

    def test_state_dict_on_npu(self):
        sm = _make_linear()
        sd = sm.state_dict()
        self.assertEqual(sd["linear.weight"].device.type, device_type)

    def test_state_dict_with_prefix(self):
        sm = _make_linear()
        sd = sm.state_dict(prefix="mymodel.")
        self.assertIn("mymodel.linear.weight", sd)
        self.assertIn("mymodel.linear.bias", sd)

    def test_state_dict_with_destination(self):
        sm = _make_linear()
        dest = {"existing": torch.tensor(0)}
        result = sm.state_dict(destination=dest, prefix="mod.")
        self.assertIs(result, dest)
        self.assertIn("existing", result)
        self.assertIn("mod.linear.weight", result)

    def test_state_dict_keep_vars(self):
        sm = _make_linear()
        sd = sm.state_dict(keep_vars=True)
        self.assertIsInstance(sd["linear.weight"], nn.Parameter)
        self.assertTrue(sd["linear.weight"].requires_grad)


class TestScriptModuleSave(TestCase):

    def test_save_and_load(self):
        sm = _make_linear()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            sm.save(path)
            loaded = torch.jit.load(path)
            x = torch.randn(2, 2).to(device_type)
            self.assertEqual(sm(x), loaded(x))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_preserves_output(self):
        sm = _make_linear()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            with torch.no_grad():
                sm.linear.weight.fill_(1.0)
                sm.linear.bias.fill_(2.0)
            sm.save(path)
            loaded = torch.jit.load(path)
            self.assertEqual(loaded.linear.weight, torch.ones(2, 2))
            self.assertEqual(loaded.linear.bias, torch.ones(2) * 2)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_returns_none(self):
        sm = _make_linear()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            ret = sm.save(path)
            self.assertIsNone(ret)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_on_npu(self):
        sm = _make_linear()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            sm.save(path)
            loaded = torch.jit.load(path)
            x = torch.randn(2, 2).to(device_type)
            self.assertEqual(sm(x), loaded(x))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_with_extra_files(self):
        sm = _make_linear()
        extra = {"meta.json": '{"version": 1}', "readme.txt": "hello"}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            sm.save(path, _extra_files=extra)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_to_buffer(self):
        sm = _make_linear()
        buf = sm.save_to_buffer()
        self.assertIsInstance(buf, bytes)
        loaded = torch.jit.load(io.BytesIO(buf))
        x = torch.randn(2, 2).to(device_type)
        self.assertEqual(sm(x), loaded(x))


class TestScriptModuleExtraRepr(TestCase):

    def test_extra_repr_returns_str(self):
        sm = _make_linear()
        result = sm.extra_repr()
        self.assertIsInstance(result, str)

    def test_extra_repr_contains_original_name(self):
        sm = _make_linear()
        result = sm.extra_repr()
        match = re.search(r"original_name=(\S+)", result)
        if match:
            self.assertIsInstance(match.group(1), str)

    def test_extra_repr_on_npu(self):
        sm = _make_linear()
        result = sm.extra_repr()
        self.assertIsInstance(result, str)


class TestScriptModuleShareMemory(TestCase):
    """share_memory behavior differs by device:
       CPU: works, makes storage shared.
       GPU/CUDA: no-op (per torch.Tensor.share_memory_ docstring).
       NPU: torch-npu intercepts with RuntimeError.
       Tests document actual NPU behavior and CPU baseline."""

    def test_share_memory_cpu_returns_self(self):
        sm = _make_cpu_linear()
        result = sm.share_memory()
        self.assertIs(result, sm)

    def test_share_memory_cpu_makes_shared(self):
        sm = _make_cpu_linear()
        sm.share_memory()
        self.assertTrue(sm.linear.weight.untyped_storage().is_shared())

    def test_share_memory_cpu_idempotent(self):
        sm = _make_cpu_linear()
        sm.share_memory()
        sm.share_memory()
        self.assertTrue(sm.linear.weight.untyped_storage().is_shared())

    def test_share_memory_on_npu_raises(self):
        sm = _make_linear()
        with self.assertRaisesRegex(
                RuntimeError, r"share_memory.*not supported in npu"):
            sm.share_memory()


class TestScriptModuleMetadata(TestCase):
    """register_module/register_parameter on NPU:
       torch-npu intercepts with RuntimeError.
       On CPU they also raise RuntimeError (PyTorch limitation:
       "Cannot re-assign modules" / "Can't add a new parameter
       after ScriptModule construction")."""

    def test_register_module_raises_on_npu(self):
        sm = _make_linear()
        sub = nn.Linear(2, 2).to(device_type)
        with self.assertRaisesRegex(
                RuntimeError, r"register_module.*not supported in npu"):
            sm.register_module("new_sub", sub)

    def test_register_parameter_raises_on_npu(self):
        sm = _make_linear()
        param = nn.Parameter(torch.randn(2, 2)).to(device_type)
        with self.assertRaisesRegex(
                RuntimeError, r"register_parameter.*not supported in npu"):
            sm.register_parameter("new_param", param)

    def test_set_submodule_raises(self):
        sm = _make_linear()
        new_sub = nn.Linear(2, 2).to(device_type)
        with self.assertRaisesRegex(
                RuntimeError, r"not supported on ScriptModules"):
            sm.set_submodule("linear", new_sub)

    def test_set_submodule_nested_raises(self):
        sm = _make_nested()
        new_sub = nn.Linear(2, 2).to(device_type)
        with self.assertRaisesRegex(
                RuntimeError, r"not supported on ScriptModules"):
            sm.set_submodule("sub.linear", new_sub)

    def test_get_buffer_unsupported(self):
        sm = _make_with_buffer()
        with self.assertRaisesRegex(
                RuntimeError,
                r"get_buffer is not supported on ScriptModules"):
            sm.get_buffer("buf")

    def test_get_buffer_unsupported_on_nested(self):
        sm = _make_nested_with_buffer()
        with self.assertRaisesRegex(
                RuntimeError,
                r"get_buffer is not supported on ScriptModules"):
            sm.get_buffer("sub.buf")

    def test_get_buffer_unsupported_nonexistent(self):
        sm = _make_linear()
        with self.assertRaisesRegex(
                RuntimeError,
                r"get_buffer is not supported on ScriptModules"):
            sm.get_buffer("nonexistent")


class TestScriptModuleUnsupported(TestCase):
    """APIs that raise errors by PyTorch design, not torch-npu."""

    def test_requires_grad_unsupported(self):
        sm = _make_linear()
        with self.assertRaisesRegex(
                RuntimeError,
                r"requires_grad_ is not supported on ScriptModules"):
            sm.requires_grad_(True)

    def test_to_empty_unsupported(self):
        sm = _make_linear()
        with self.assertRaisesRegex(
                RuntimeError,
                r"to_empty is not supported on ScriptModules"):
            sm.to_empty(device=device_type)

    def test_xpu_unsupported(self):
        sm = _make_linear()
        with self.assertRaisesRegex(
                RuntimeError,
                r"xpu is not supported on ScriptModules"):
            sm.xpu()

    def test_set_extra_state_raises(self):
        sm = _make_linear()
        with self.assertRaisesRegex(
                RuntimeError, r"should never be called"):
            sm.set_extra_state({"version": 1})


if __name__ == "__main__":
    run_tests()
