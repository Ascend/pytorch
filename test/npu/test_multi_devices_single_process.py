import os
import torch
import numpy as np

import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:False'


class TestNpu(TestCase):

    @skipIfUnsupportMultiNPU(2)
    def test_creat_tensor(self):
        device_number = torch.npu.device_count()
        for device_idx in range(device_number):
            t = torch.randn(2, 255, 255, device=f"npu:{device_idx}")
            self.assertTrue(str(t.device) == f"npu:{device_idx}")

    @skipIfUnsupportMultiNPU(2)
    def test_creat_stream(self):
        device_number = torch.npu.device_count()
        stream_instance = set()
        for i in range(device_number):
            torch.npu.set_device(i)
            default_stream = torch.npu.default_stream()
            current_stream = torch.npu.current_stream()
            stream_instance.add(current_stream)
        self.assertTrue(len(stream_instance) == device_number)

    def _test_host_to_device(self, t_cpu):
        t_device_0 = t_cpu.to("npu:0")
        self.assertTrue(str(t_device_0.device) == "npu:0")
        t_device_1 = t_cpu.to("npu:1")
        self.assertTrue(str(t_device_1.device) == "npu:1")
        self.assertRtolEqual(t_cpu.numpy(), t_device_0.cpu().numpy())
        self.assertRtolEqual(t_cpu.numpy(), t_device_1.cpu().numpy())

    def _test_device_to_device(self, t_cpu):
        t_device_0 = t_cpu.to("npu:0")
        self.assertTrue(str(t_device_0.device) == "npu:0")
        t_device_1 = t_device_0.to("npu:1")
        self.assertTrue(str(t_device_1.device) == "npu:1")
        self.assertRtolEqual(t_device_1.cpu(
        ).numpy(), t_device_0.cpu().numpy())

    def _test_device_copy(self):
        t_cpu = torch.rand(2, 255, 255)
        self._test_host_to_device(t_cpu)
        self._test_device_to_device(t_cpu)

    def _test_module(self):

        class MyModule(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 2)
                self.fc2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                if (x.device.type == 'cpu'):
                    x = self.fc1(x)
                    return self.fc2(x)
                else:
                    self.fc1 = self.fc1.to("npu:0")
                    self.fc2 = self.fc2.to("npu:1")
                    x = self.fc1(x)
                    return self.fc2(x.to("npu:1"))

        module = MyModule()
        module.eval()
        input_t = torch.rand(8, 2)
        output_cpu = module(input_t)
        output_npu = module(input_t.to("npu:0"))
        self.assertTrue(str(output_npu.device) == "npu:1")
        self.assertRtolEqual(output_cpu.detach().numpy(), output_npu.detach().cpu().numpy(), prec=1.e-3)

    @skipIfUnsupportMultiNPU(2)
    def test_aclop_with_multi_device(self):
        torch.npu.set_compile_mode(jit_compile=True)
        self._test_device_copy()
        self._test_module()

    @skipIfUnsupportMultiNPU(2)
    def test_opapi_with_multi_device(self):
        torch.npu.set_compile_mode(jit_compile=False)
        self._test_device_copy()
        self._test_module()


class TestOp(TestCase):

    def _cpu_op_exec(self, input1):
        output = torch.abs(input1)
        output = output.cpu().numpy()
        return output

    def _npu_op_exec(self, input1):
        output = torch.abs(input1)
        output = output.cpu().numpy()
        return output
    
    def _test_abs(self, device="npu:1"):
        torch.npu.set_device(0)
        cpu_input = torch.Tensor([1, -2, -10])
        npu_input = cpu_input.to(device)
        cpu_output = self._cpu_op_exec(cpu_input)
        npu_output = self._npu_op_exec(npu_input)
        self.assertRtolEqual(cpu_output, npu_output)

    def _test_isfinite(self, device="npu:1"):
        torch.npu.set_device(0)
        x = torch.Tensor([1, 2, -10]).to(device)
        output = torch.isfinite(x)
        self.assertTrue(output.all())

    def _test_unique_dim(self, device="npu:1", dtype=torch.float):
        torch.npu.set_device(0)
        self.assertFalse(hasattr(torch, "unique_dim"))

        x = torch.tensor([[[1., 1.],
                               [0., 1.],
                               [2., 1.],
                               [0., 1.]],
                              [[1., 1.],
                               [0., 1.],
                               [2., 1.],
                               [0., 1.]]],
                             dtype=dtype,
                             device=device)
        expected_unique_dim0 = torch.tensor([[[1., 1.],
                                                  [0., 1.],
                                                  [2., 1.],
                                                  [0., 1.]]],
                                                dtype=dtype,
                                                device=device)
        expected_inverse_dim0 = torch.tensor([0, 0])
        expected_counts_dim0 = torch.tensor([2])

        x_unique, x_inverse, x_counts = torch.unique(
                x,
                return_inverse=True,
                return_counts=True,
                dim=0)
        self.assertEqual(expected_unique_dim0, x_unique)
        self.assertEqual(expected_inverse_dim0, x_inverse)
        self.assertEqual(expected_counts_dim0, x_counts)

    def _supported_op_exec(self, query_states1, past_key, past_value, head_dim):
        attn_weights1 = torch.matmul(query_states1, past_key.transpose(2, 3)) / 0.0078125
        attn_weights1 = torch.max(attn_weights1, torch.full(
            (1, 1), torch.finfo(attn_weights1.dtype).min, device=attn_weights1.device))
        attn_weights1 = torch.nn.functional.softmax(attn_weights1, dim=-1, dtype=torch.float32).to(query_states1.dtype)
        attn_output1 = torch.matmul(attn_weights1, past_value)
        return attn_output1

    def _custom_op_exec(self, query, key, value, head_dim):
        scale = 1 / 0.0078125
        return torch_npu.npu_prompt_flash_attention(
            query, key, value, num_heads=32, input_layout="BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535, sparse_mode=0)
    
    @SupportedDevices(['Ascend910B'])
    def _test_npu_prompt_flash_attention(self, device="npu:1"):
        torch.npu.set_device(0)
        query = torch.randn(1, 32, 2048, 128, dtype=torch.float16).to(device)
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).to(device)
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).to(device)

        head_dim = 128

        supported_output = self._supported_op_exec(query, key, value, head_dim)
        custom_output = self._custom_op_exec(query, key, value, head_dim)
        self.assertRtolEqual(supported_output, custom_output)

    @skipIfUnsupportMultiNPU(2)
    def test_aclop_op_with_multi_device(self):
        torch.npu.set_compile_mode(jit_compile=True)
        self._test_abs()
        self._test_isfinite()
        self._test_unique_dim()
        self._test_npu_prompt_flash_attention()

    @skipIfUnsupportMultiNPU(2)
    def test_opapi_op_with_multi_device(self):
        torch.npu.set_compile_mode(jit_compile=False)
        self._test_abs()
        self._test_isfinite()
        self._test_unique_dim()
        self._test_npu_prompt_flash_attention()


if __name__ == '__main__':
    run_tests()
