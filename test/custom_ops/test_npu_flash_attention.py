import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestNPUFlashAttention(TestCase):
    def supported_op_exec(self, query, key, value):
        scale = 0.08838
        qk = torch.matmul(query, key.transpose(2, 3)).mul(scale)
        softmax_res = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(torch.float16)
        output = torch.matmul(softmax_res, value)
        output = output.transpose(1, 2)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        return output

    def custom_op_exec(self, query, key, value):
        scale = 0.08838
        return torch_npu.npu_fusion_attention(
            query, key, value, head_num=32, input_layout="BSH", scale=scale)

    def trans_BNSD2BSH(self, tensor: torch.Tensor):
        tensor = torch.transpose(tensor, 1, 2)
        tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
        return tensor

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', 'device type is not supported, skip this ut!')
    def test_npu_flash_attention(self, device="npu"):
        query = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        key = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        value = torch.randn(1, 32, 128, 128, dtype=torch.float16)

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()
        output = self.supported_op_exec(query, key, value)
        attention_score, softmax_max, softmax_sum, softmax_out, seed, offset, numels = self.custom_op_exec(q_npu, k_npu, v_npu)
        self.assertRtolEqual(output, attention_score, prec=0.005, prec16=0.005)

if __name__ == "__main__":
    run_tests()

