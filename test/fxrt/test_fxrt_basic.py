# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Licensed under the BSD 3-Clause License.

import importlib.util
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
        from torch_npu.fxrt import backend

        torch.manual_seed(0)
        model = SimpleMLP().npu().eval()
        x = torch.randn(4, 16).npu()
        compiled = torch.compile(model, backend=backend)
        with torch.no_grad():
            expected = model(x)
            # The first call compiles the graph for fxrt; the second runs the cached graph.
            for _ in range(2):
                self.assertRtolEqual(expected.cpu().numpy(), compiled(x).cpu().numpy())


if __name__ == "__main__":
    run_tests()
