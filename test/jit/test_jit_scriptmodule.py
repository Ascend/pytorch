"""
Add validation cases for torch.jit.ScriptModule APIs on NPU:

PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
This file validates torch.jit.ScriptModule.bfloat16, torch.jit.ScriptModule.buffers,
torch.jit.ScriptModule.children, torch.jit.ScriptModule.code
torch.jit.ScriptModule.compile, torch.jit.ScriptModule.cpu
"""

import torch
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

class TestJitScriptModuleCPU(TestCase):
    def _check_model_device(self, model: torch.jit.ScriptModule, device_type: str):
        """Check whether all parameters and buffers of the model are on the specified device."""
        for param in model.parameters():
            self.assertEqual(param.device.type, device_type)
        for buf in model.buffers():
            self.assertEqual(buf.device.type, device_type)

    def _run_infer(self, model: torch.jit.ScriptModule, device):
        x = torch.randn(2, 1, 32, 32).to(device)
        out = model(x)
        self.assertEqual(out.dim(), 4)

    def test_script_module_cpu_from_npu(self):
        # Skip test if NPU is not available
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            self.skipTest("NPU device unavailable, skip this test case")

        # Instantiate model and convert to ScriptModule
        model = Model()
        script_model = torch.jit.script(model)

        # Move ScriptModule to NPU and verify device
        script_model = script_model.npu()
        self._check_model_device(script_model, "npu")

        # Transfer model from NPU back to CPU via cpu() method
        cpu_script_model = script_model.cpu()

        # Assert model type, device and normal inference execution
        self.assertIsInstance(cpu_script_model, torch.jit.ScriptModule)
        self._check_model_device(cpu_script_model, "cpu")
        self._run_infer(cpu_script_model, torch.device("cpu"))

class TestJitScriptModuleCompile(TestCase):
    def _check_model_device(self, model: torch.jit.ScriptModule, device_type: str):
        """Check whether all parameters and buffers of the model are on the specified device."""
        for param in model.parameters():
            self.assertEqual(param.device.type, device_type)
        for buf in model.buffers():
            self.assertEqual(buf.device.type, device_type)

    def _run_infer(self, model, device):
        """Run model inference and verify output tensor shape validity."""
        x = torch.randn(2, 1, 32, 32).to(device)
        out = model(x)
        self.assertEqual(out.dim(), 4)

    def test_script_module_compile(self):
        """Test compilation and inference using torch.compile with NPU backend."""
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            self.skipTest("NPU device is unavailable, skip this test case")
        torch.npu.config.allow_internal_format = False

        # Initialize model and convert to ScriptModule
        model = Model()
        scripted_model = torch.jit.script(model).npu()

        # Move model to NPU and check device placement
        self._check_model_device(scripted_model, "npu")

        # Compile model with NPU backend
        compiled_model = torch.compile(scripted_model, backend="npugraph_ex", fullgraph=True, dynamic=False)

        # run inference validation
        self._run_infer(compiled_model, torch.device("npu"))

if __name__ == "__main__":
    run_tests()
