"""
Add validation cases for torch.jit.ScriptModule APIs on NPU:

PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
This file validates torch.jit.ScriptModule.bfloat16, torch.jit.ScriptModule.buffers, torch.jit.ScriptModule.children, torch.jit.ScriptModule.code
"""

import torch
import torch_npu
from torch.testing._internal.common_utils import TestCase, run_tests
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

        # Register buffer
        self.register_buffer('register_buffer_out1', torch.randn(20))
        self.register_buffer('register_buffer_out2', torch.randn(20, 1, 5, 5))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


class TestJitScriptModuleBfloat16(TestCase):
    def test_bfloat16(self):
        model = Model()
        uninit_param = torch.nn.parameter.UninitializedParameter(device="npu")

        self.assertEqual(uninit_param.device.type, "npu")
        self.assertIsInstance(model, nn.Module)

        model_bf16 = model.bfloat16()
        self.assertIsInstance(model_bf16, nn.Module)
        # The validation parameters have been converted to bfloat16
        for param in model_bf16.parameters():
            self.assertEqual(param.dtype, torch.bfloat16)

    def test_jit_bfloat16(self):
        model = Model().bfloat16()
        jit_model = torch.jit.script(model)
        uninit_param = torch.nn.parameter.UninitializedParameter(device="npu")

        self.assertEqual(uninit_param.device.type, "npu")
        self.assertIsInstance(jit_model, torch.jit.ScriptModule)
        # Verify the feasibility of JIT model execution
        dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.bfloat16, device="cpu")
        output = jit_model(dummy_input)
        self.assertEqual(output.dtype, torch.bfloat16)


class TestJitScriptModuleBuffers(TestCase):
    def test_buffers(self):
        model = Model()
        uninit_param = torch.nn.parameter.UninitializedParameter(device="npu")

        self.assertEqual(uninit_param.device.type, "npu")
        buffers = list(model.buffers())
        self.assertEqual(len(buffers), 2)  # Two registered buffers
        self.assertTrue(all(isinstance(b, torch.Tensor) for b in buffers))

    def test_jit_buffers(self):
        model = torch.jit.script(Model())
        uninit_param = torch.nn.parameter.UninitializedParameter(device="npu")

        self.assertEqual(uninit_param.device.type, "npu")
        buffers = list(model.buffers())
        self.assertEqual(len(buffers), 2)
        self.assertTrue(all(isinstance(b, torch.Tensor) for b in buffers))


class TestJitScriptModuleChildren(TestCase):
    def test_children(self):
        model = Model()
        uninit_param = torch.nn.parameter.UninitializedParameter(device="npu")

        self.assertEqual(uninit_param.device.type, "npu")
        children = list(model.children())
        self.assertEqual(len(children), 2)  # conv1, conv2
        self.assertTrue(all(isinstance(c, nn.Module) for c in children))

    def test_jit_children(self):
        model = torch.jit.script(Model())
        uninit_param = torch.nn.parameter.UninitializedParameter(device="npu")

        self.assertEqual(uninit_param.device.type, "npu")
        children = list(model.children())
        self.assertEqual(len(children), 2)
        self.assertTrue(all(isinstance(c, torch.nn.Module) for c in children))


class TestJitScriptModuleCode(TestCase):
    def test_jit_code(self):
        model = torch.jit.script(Model())
        uninit_param = torch.nn.parameter.UninitializedParameter(device="npu")

        self.assertEqual(uninit_param.device.type, "npu")
        self.assertTrue(hasattr(model, 'code'))
        self.assertIn('forward', str(model.code))


if __name__ == "__main__":
    run_tests()
