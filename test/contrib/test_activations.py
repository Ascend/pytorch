import unittest
import numpy as np
import torch
import torch.nn.functional as F

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.module import Mish, SiLU
from torch_npu.contrib.function.fused_attention import _check_compatibility_once
from torch_npu.contrib.function.fused_attention import _is_format_matched


class TestActivations(TestCase):

    def cpu_mish(self, input1):
        """
        Applies the mish function element-wise:
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        See additional documentation for mish class.
        """
        input1.requires_grad = True
        res = input1 * torch.tanh(F.softplus(input1))
        res.backward(torch.ones_like(res))
        return res.detach(), input1.grad

    def npu_mish(self, input1):
        input1.requires_grad = True
        model = Mish()
        res = model(input1)
        res.backward(torch.ones_like(res))
        return res.detach().cpu(), input1.grad.cpu()
    
    def test_mish(self):
        dtype_list = [np.float16, np.float32]
        format_list = [-1, 0, 2]
        shape_list = [
            [4],
            [2, 3],
            [6, 5, 8, 10],
            [1, 2, 3, 6, 6],
            [2, 5, 6, 8, 9, 2],
            [2, 5, 6, 8, 9, 2, 2],
        ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.float()
                cpu_output, cpu_inputgrad = self.cpu_mish(cpu_input)
                cpu_output = cpu_output.half()
                cpu_inputgrad = cpu_inputgrad.half()
            else:
                cpu_output, cpu_inputgrad = self.cpu_mish(cpu_input)

            npu_output, npu_inputgrad = self.npu_mish(npu_input)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inputgrad, npu_inputgrad)

    def cpu_silu(self, input1):
        """
        Applies the mish function element-wise:
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        See additional documentation for mish class.
        """
        input1.requires_grad = True
        res = input1 * torch.sigmoid(input1)
        output = res.sum()
        output.backward()
        return res.detach(), input1.grad

    def npu_silu(self, input1):
        input1.requires_grad = True
        model = SiLU()
        res = model(input1)
        output = res.sum()
        output.backward()
        return res.detach().cpu(), input1.grad.cpu()

    def test_silu(self):
        dtype_list = [np.float32, np.float16]
        format_list = [-1, 0, 2]
        shape_list = [
            [5],
            [2, 3],
            [6, 5, 2, 10],
            [1, 2, 4, 6, 6],
            [2, 5, 6, 2, 9, 2],
            [2, 5, 6, 3, 9, 2, 2],
        ]

        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.float()
                cpu_output, cpu_inputgrad = self.cpu_silu(cpu_input)
                cpu_output = cpu_output.half()
                cpu_inputgrad = cpu_inputgrad.half()
            else:
                cpu_output, cpu_inputgrad = self.cpu_silu(cpu_input)

            npu_output, npu_inputgrad = self.npu_silu(npu_input)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inputgrad, npu_inputgrad)

    def test_check_compatibility_once_invalid_hidden_states_shape(self):
        hidden_states = torch_npu.npu_format_cast(torch.randn(30, 1024).npu(), 29)
        attention_mask = torch_npu.npu_format_cast(torch.randn(2, 1, 8, 8).npu(), 29)
        query_kernel = torch_npu.npu_format_cast(torch.randn(1024, 1024).npu(), 29)
        key_kernel = torch_npu.npu_format_cast(torch.randn(1024, 1024).npu(), 29)
        value_kernel = torch_npu.npu_format_cast(torch.randn(1024, 1024).npu(), 29)
        query_bias = torch_npu.npu_format_cast(torch.randn(1024).npu(), 2)
        key_bias = torch_npu.npu_format_cast(torch.randn(1024).npu(), 2)
        value_bias = torch_npu.npu_format_cast(torch.randn(1024).npu(), 2)

        with self.assertRaises(RuntimeError):
            _check_compatibility_once(
                hidden_states,
                attention_mask,
                query_kernel,
                key_kernel,
                value_kernel,
                query_bias,
                key_bias,
                value_bias
            )

    def test_is_format_matched_invalid(self):
        tensor1 = torch_npu.npu_format_cast(torch.randn(4, 4).npu(), 29)
        tensor2 = torch_npu.npu_format_cast(torch.randn(4, 4).npu(), 29)
        tensor3 = torch_npu.npu_format_cast(torch.randn(4, 4).npu(), 29)
        tensor4 = torch_npu.npu_format_cast(torch.randn(4, 4).npu(), 29)
        tensor5 = torch_npu.npu_format_cast(torch.randn(4, 4).npu(), 29)
        tensor6 = torch_npu.npu_format_cast(torch.randn(4, 4).npu(), 2)
        tensor7 = torch_npu.npu_format_cast(torch.randn(4, 4).npu(), 2)
        tensor8 = torch_npu.npu_format_cast(torch.randn(4, 4).npu(), 29)

        result = _is_format_matched([tensor1, tensor2, tensor3, tensor4, tensor5, tensor6, tensor7, tensor8])
        self.assertFalse(result)


if __name__ == "__main__":
    run_tests()
