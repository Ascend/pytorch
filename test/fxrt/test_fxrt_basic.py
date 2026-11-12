# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Licensed under the BSD 3-Clause License.

import importlib.util
import subprocess
import sys
import unittest

import torch

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

    def test_backend_discovery_is_lazy(self):
        probe = '''
import sys
import torch
import torch_npu
from torch._dynamo.backends.registry import lookup_backend

assert "fxrt" in torch._dynamo.list_backends()
assert "fxrt" not in sys.modules
assert "torch_npu.fxrt" not in sys.modules
backend = lookup_backend("fxrt")
from torch_npu.fxrt.fx_backend import backend as implementation
import torch_npu.fxrt as fxrt
assert backend is implementation
assert lookup_backend("fxrt") is backend
assert "backend" not in fxrt.__all__
assert not hasattr(fxrt, "backend")
assert not torch_npu.npu.is_initialized()
'''
        proc = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=False
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    run_tests()
