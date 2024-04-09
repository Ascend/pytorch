import torch
import torch.nn as nn

from torch_npu.testing.testcase import TestCase, run_tests


class TorchBackendsApiTestCase(TestCase):

    def test_flash_sdp(self):
        torch.npu.enable_flash_sdp(True)
        res = torch.npu.flash_sdp_enabled()
        self.assertEqual(res, True)
        torch.npu.enable_flash_sdp(False)
        res = torch.npu.flash_sdp_enabled()
        self.assertEqual(res, False)

    def test_mem_efficient_sdp(self):
        torch.npu.enable_mem_efficient_sdp(False)
        res = torch.npu.mem_efficient_sdp_enabled()
        self.assertEqual(res, False)
        torch.npu.enable_mem_efficient_sdp(True)
        res = torch.npu.mem_efficient_sdp_enabled()
        self.assertEqual(res, True)

    def test_math_sdp(self):
        torch.npu.enable_math_sdp(True)
        res = torch.npu.math_sdp_enabled()
        self.assertEqual(res, True)
        torch.npu.enable_math_sdp(False)
        res = torch.npu.math_sdp_enabled()
        self.assertEqual(res, False)

    def test_sdp_kernel(self):
        torch.npu.enable_flash_sdp(False)
        torch.npu.enable_mem_efficient_sdp(False)
        torch.npu.enable_math_sdp(False)
        torch.npu.sdp_kernel()
        flash_res = torch.npu.flash_sdp_enabled()
        mem_mem_efficient_res = torch.npu.mem_efficient_sdp_enabled()
        math_res = torch.npu.math_sdp_enabled()
        self.assertEqual(flash_res, False)
        self.assertEqual(mem_mem_efficient_res, False)
        self.assertEqual(math_res, False)

    def test_aclnn_allow_hf32(self):
        res = torch.npu.aclnn.allow_hf32
        self.assertEqual(res, True)
        with torch.npu.aclnn.flags(allow_hf32=True):
            res = torch.npu.aclnn.allow_hf32
            self.assertEqual(res, True)
        with torch.npu.aclnn.flags(allow_hf32=False):
            res = torch.npu.aclnn.allow_hf32
            self.assertEqual(res, False)

    def test_conv_allow_hf32(self):
        torch.npu.conv.allow_hf32 = True
        res = torch.npu.conv.allow_hf32
        self.assertEqual(res, True)
        torch.npu.conv.allow_hf32 = False
        res = torch.npu.conv.allow_hf32
        self.assertEqual(res, False)

    def test_aclnn_to_conv_allow_hf32(self):
        torch.npu.conv.allow_hf32 = True
        res = torch.npu.aclnn.allow_hf32
        self.assertEqual(res, True)
        torch.npu.conv.allow_hf32 = False
        res = torch.npu.aclnn.allow_hf32
        self.assertEqual(res, False)
        with torch.npu.aclnn.flags(allow_hf32=True):
            res = torch.npu.conv.allow_hf32
            self.assertEqual(res, True)
        with torch.npu.aclnn.flags(allow_hf32=False):
            res = torch.npu.conv.allow_hf32
            self.assertEqual(res, False)

    def test_matmul_allow_hf32(self):
        res = torch.npu.matmul.allow_hf32
        self.assertEqual(res, False)
        torch.npu.matmul.allow_hf32 = True
        res = torch.npu.matmul.allow_hf32
        self.assertEqual(res, True)
        torch.npu.matmul.allow_hf32 = False
        res = torch.npu.matmul.allow_hf32
        self.assertEqual(res, False)

    def test_preferred_linalg_library(self):
        res = torch.npu.preferred_linalg_library()
        self.assertEqual(res, torch._C._LinalgBackend.Default)
        res = torch.npu.preferred_linalg_library("Cusolver")
        self.assertEqual(res, torch._C._LinalgBackend.Default)
        res = torch.npu.preferred_linalg_library(torch._C._LinalgBackend.Magma)
        self.assertEqual(res, torch._C._LinalgBackend.Default)

    def test_enable_deterministic_with_backward(self):
        target_dtype = torch.float16

        class DeterministicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + 1
                x = torch.npu.enable_deterministic_with_backward(x)
                add1 = x + y
                add1 = sum(add1)
                add1 = add1 + add1
                return add1

        device = torch.device("npu:0")
        model = DeterministicModel()
        npu_mode = model.to(device)
        ins1 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        ins2 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        output_data = npu_mode(ins1, ins2)
        self.assertEqual(True, torch.are_deterministic_algorithms_enabled())
        loss_fn = nn.MSELoss()
        target_data = torch.randn((1, 2), requires_grad=True).to(target_dtype).npu()
        loss = loss_fn(output_data, target_data)
        loss.backward()
        self.assertEqual(False, torch.are_deterministic_algorithms_enabled())

    def test_disable_deterministic_with_backward(self):
        target_dtype = torch.float16

        class DeterministicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + 1
                x = torch.npu.disable_deterministic_with_backward(x)
                add1 = x + y
                add1 = sum(add1)
                add6 = add1 + add1
                return add6

        device = torch.device("npu:0")
        model = DeterministicModel()
        npu_mode = model.to(device)
        ins1 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        ins2 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        output_data = npu_mode(ins1, ins2)
        self.assertEqual(False, torch.are_deterministic_algorithms_enabled())
        loss_fn = nn.MSELoss()
        target_data = torch.randn((1, 2), requires_grad=True).to(target_dtype).npu()
        loss = loss_fn(output_data, target_data)
        loss.backward()
        self.assertEqual(True, torch.are_deterministic_algorithms_enabled())

    def test_enable_to_disable_deterministic_with_backward(self):
        target_dtype = torch.float16

        class DeterministicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + 1
                x = torch.npu.enable_deterministic_with_backward(x)
                add4 = x + y
                add1 = sum(add4)
                add1 = torch.npu.disable_deterministic_with_backward(add1)
                add6 = add1 + add1
                return add6

        device = torch.device("npu:0")
        model = DeterministicModel()
        npu_mode = model.to(device)
        ins1 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        ins2 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        output_data = npu_mode(ins1, ins2)
        self.assertEqual(False, torch.are_deterministic_algorithms_enabled())
        loss_fn = nn.MSELoss()
        target_data = torch.randn((1, 2), requires_grad=True).to(target_dtype).npu()
        loss = loss_fn(output_data, target_data)
        loss.backward()
        self.assertEqual(False, torch.are_deterministic_algorithms_enabled())

    def test_disable_to_endable_deterministic_algorithms(self):
        target_dtype = torch.float16

        class DeterministicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + 1
                x = torch.npu.disable_deterministic_with_backward(x)
                add4 = x + y
                add1 = sum(add4)
                add1 = torch.npu.enable_deterministic_with_backward(add1)
                add6 = add1 + add1
                return add6

        device = torch.device("npu:0")
        model = DeterministicModel()
        npu_mode = model.to(device)
        ins1 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        ins2 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        output_data = npu_mode(ins1, ins2)
        self.assertEqual(True, torch.are_deterministic_algorithms_enabled())
        loss_fn = nn.MSELoss()
        target_data = torch.randn((1, 2), requires_grad=True).to(target_dtype).npu()
        loss = loss_fn(output_data, target_data)
        loss.backward()
        self.assertEqual(True, torch.are_deterministic_algorithms_enabled())

    def test_endable_deterministic_algorithms_in_graph(self):
        try:
            import torchair as tng
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            npu_backend = tng.get_npu_backend(compiler_config=config)
        except Exception:
            return True
        target_dtype = torch.float16

        class DeterministicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + 1
                x = torch.npu.disable_deterministic_with_backward(x)
                add4 = x + y
                add1 = sum(add4)
                add1 = torch.npu.enable_deterministic_with_backward(add1)
                add6 = add1 + add1
                return add6

        device = torch.device("npu:0")
        model = DeterministicModel()
        npu_mode = model.to(device)
        ins1 = torch.ones((2, 2)).to(target_dtype).npu()
        ins2 = torch.ones((2, 2)).to(target_dtype).npu()
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        with self.assertRaises(AssertionError):
            npu_mode(ins1, ins2)

    def test_endable_deterministic_algorithms_in_graphbyautograd(self):
        try:
            import torchair as tng
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            npu_backend = tng.get_npu_backend(compiler_config=config)
        except Exception:
            return True
        target_dtype = torch.float16

        class DeterministicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + 1
                x = torch.npu.disable_deterministic_with_backward(x)
                add4 = x + y
                add1 = sum(add4)
                add1 = torch.npu.enable_deterministic_with_backward(add1)
                add6 = add1 + add1
                return add6

        device = torch.device("npu:0")
        model = DeterministicModel()
        npu_mode = model.to(device)
        ins1 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        ins2 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        with self.assertRaises(AssertionError):
            npu_mode(ins1, ins2)


if __name__ == "__main__":
    run_tests()
