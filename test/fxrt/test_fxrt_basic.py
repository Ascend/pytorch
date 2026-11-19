# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Licensed under the BSD 3-Clause License.

import importlib.util
import subprocess
import sys
import unittest
from unittest import mock

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize

import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests


# Look the package up without importing it: importing torch_npu does not load fxrt.
_HAS_FXRT = importlib.util.find_spec("torch_npu.fxrt") is not None


class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@unittest.skipUnless(_HAS_FXRT, "torch_npu is built without fxrt")
class TestFxrtBasic(TestCase):
    """Run a basic model through the fxrt torch.compile backend on NPU."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def test_compiled_mlp_matches_eager(self):
        torch.manual_seed(0)
        model = SimpleMLP().npu().eval()
        x = torch.randn(4, 16).npu()
        compiled = torch.compile(model, backend="fxrt")
        with torch.no_grad():
            expected = model(x)
            # The first call compiles the graph for fxrt; the second runs the cached graph.
            for _ in range(2):
                self.assertRtolEqual(expected.cpu().numpy(), compiled(x).cpu().numpy())

    @parametrize("import_order", ("entry_point_first", "fxrt_first"))
    def test_backend_discovery_is_lazy(self, import_order):
        probe = '''
import sys
from importlib.util import find_spec
from pathlib import Path

import torch
import torch_npu
from torch._dynamo.backends.registry import lookup_backend

assert "fxrt" in torch._dynamo.list_backends()
assert "fxrt" not in sys.modules
assert "torch_npu.fxrt" not in sys.modules
if sys.argv[1] == "fxrt_first":
    # Make the installed package importable as fxrt, as in a source checkout.
    package_dir = Path(find_spec("torch_npu.fxrt").origin).parent
    sys.path.insert(0, str(package_dir.parent))
    import fxrt
    sys.path.pop(0)
    assert "torch_npu.fxrt" not in sys.modules

backend = lookup_backend("fxrt")
from torch_npu.fxrt.fx_backend import backend as implementation
import torch_npu.fxrt as fxrt
assert backend is implementation
assert lookup_backend("fxrt") is backend
assert fxrt is sys.modules["fxrt"]
assert sys.modules["torch_npu.fxrt.fx_wrapper"] is sys.modules["fxrt.fx_wrapper"]
assert not fxrt.__all__
assert not hasattr(fxrt, "backend")
assert not hasattr(fxrt, "ops")
assert not hasattr(fxrt, "__version__")
assert not torch_npu.npu.is_initialized()
'''
        proc = subprocess.run(
            [sys.executable, "-c", probe, import_order], capture_output=True, text=True, check=False
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    @parametrize("configuration", ("options", "patch", "assignment"))
    def test_inductor_fxrt_config_is_captured(self, configuration):
        from torch._inductor import config
        from torch._inductor.codegen.common import device_codegens
        from torch_npu.fxrt.fx_wrapper import FxrtFxWrapper

        def model(x):
            return torch.mm(x, x)

        x = torch.randn(8, 8, device="npu")
        expected = model(x)
        if configuration == "options":
            compiled = torch.compile(
                model, backend="inductor", fullgraph=True, options={"fxrt": True}
            )
        else:
            import torch_npu._inductor  # noqa: F401

            if configuration == "patch":
                with config.patch({"fxrt": True}):
                    compiled = torch.compile(model, backend="inductor", fullgraph=True)
            else:
                previous_fxrt = config.fxrt
                config.fxrt = True
                try:
                    compiled = torch.compile(model, backend="inductor", fullgraph=True)
                finally:
                    config.fxrt = previous_fxrt
        plain = torch.compile(model, backend="inductor", fullgraph=True)

        with mock.patch.object(
            FxrtFxWrapper, "compile_graph", autospec=True, side_effect=FxrtFxWrapper.compile_graph
        ) as compile_graph:
            # Check the default Inductor output and wrapper.
            torch.testing.assert_close(plain(x), expected)
            compile_graph.assert_not_called()
            previous_codegen = device_codegens["npu"]

            # Check that FXRT is selected and matches eager.
            torch.testing.assert_close(compiled(x), expected)
            compile_graph.assert_called_once()
            # Check that compilation restored the original codegen.
            self.assertIs(device_codegens["npu"], previous_codegen)
        # Check that the FXRT setting returned to its default.
        self.assertIs(config.fxrt, False)

    @parametrize("failure", ("entry", "cache"))
    def test_inductor_fxrt_config_failure_preserves_plain_inductor(self, failure):
        probe = r'''
import sys
from unittest import mock

import torch
import torch_npu
from torch._inductor import config
from torch.utils import _config_module

assert "fxrt" not in config._config
original_entry = _config_module._ConfigEntry
original_mark_dirty = type(config)._mark_get_dict_dirty

def make_entry(*args, **kwargs):
    if kwargs.get("name") == "fxrt":
        raise TypeError("Simulated config entry API change")
    return original_entry(*args, **kwargs)

def mark_dirty(self, name):
    if self is config and name == "fxrt":
        raise AttributeError("Simulated config cache API change")
    return original_mark_dirty(self, name)

target, name, replacement = (
    (_config_module, "_ConfigEntry", make_entry)
    if sys.argv[1] == "entry"
    else (type(config), "_mark_get_dict_dirty", mark_dirty)
)

def model(x):
    return torch.mm(x, x)

x = torch.randn(8, 8, device="npu")
expected = model(x)
with mock.patch.object(target, name, replacement):
    for options in (
        None,
        {"max_autotune": False},
        {"fxrt": False},
        {"fxrt": False, "max_autotune": False},
    ):
        compiled = torch.compile(model, backend="inductor", options=options, fullgraph=True)
        torch.testing.assert_close(compiled(x), expected)
        torch._dynamo.reset()
    assert "fxrt" not in config.get_config_copy()
    assert "torch_npu.fxrt" not in sys.modules
    try:
        torch.compile(model, backend="inductor", options={"fxrt": True})
    except RuntimeError as error:
        assert "FXRT configuration is unavailable" in str(error), str(error)
    else:
        raise AssertionError("FXRT must report the unsupported configuration API")
'''
        proc = subprocess.run(
            [sys.executable, "-c", probe, failure], capture_output=True, text=True, check=False
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


instantiate_parametrized_tests(TestFxrtBasic)


if __name__ == "__main__":
    run_tests()
