import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


class TestNPUFlashAttention(TestCase):
    def supported_op_exec(self, query, key, value, dy):
        scale = 0.08838
        qk = torch.matmul(query, key.transpose(2, 3)).mul(scale)
        softmax_res, x_max, x_sum = tsoftmax(qk.to(torch.float32))
        softmax_res = softmax_res.to(torch.float16)
        y = torch.matmul(softmax_res, value)
        dv = torch.matmul(softmax_res.transpose(2, 3), dy)
        dp = torch.matmul(dy, value.transpose(2, 3))
        softmax_grad_res = (tsoftmax_grad(dp, softmax_res) * scale)
        dq = torch.matmul(softmax_grad_res, key)
        dk = torch.matmul(softmax_grad_res.transpose(2, 3), query)       
        dq = dq.transpose(1, 2)
        dq = dq.reshape(dq.shape[0], dq.shape[1], -1)
        dk = dk.transpose(1, 2)
        dk = dk.reshape(dk.shape[0], dk.shape[1], -1)
        dv = dv.transpose(1, 2)
        dv = dv.reshape(dv.shape[0], dv.shape[1], -1)
        return y, softmax_res, x_max, x_sum, dq, dk, dv

    def custom_op_exec(self, query, key, value, dy, softmax_max, softmax_sum, attention_in):
        scale = 0.08838
        return torch_npu.npu_fusion_attention_grad(
            query, key, value, dy, head_num=32, input_layout="BSH",
            softmax_max=softmax_max, softmax_sum=softmax_sum, attention_in=attention_in, scale_value=scale)

    def trans_BNSD2BSH(self, tensor: torch.Tensor):
        tensor = torch.transpose(tensor, 1, 2)
        tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
        return tensor

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', 'device type is not supported, skip this ut!')
    def test_npu_flash_attention(self, device="npu"):
        query = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        key = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        value = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        dy = torch.randn(1, 32, 128, 128, dtype=torch.float16)

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()
        dy_npu = self.trans_BNSD2BSH(dy).npu()
        out, softmax_res, x_max, x_sum, dq_cpu, dk_cpu, dv_cpu = self.supported_op_exec(query, key, value, dy)
        x_max = x_max.expand(1, 32, 128, 8).npu()
        x_sum = x_sum.expand(1, 32, 128, 8).npu()
        out_npu = self.trans_BNSD2BSH(out).npu()
        dq, dk, dv, dpse = self.custom_op_exec(q_npu, k_npu, v_npu, dy_npu, x_max, x_sum, out_npu)
        print(dq_cpu.shape)
        print(dq.shape)
        self.assertRtolEqual(dq_cpu, dq, prec=0.005, prec16=0.005)

if __name__ == "__main__":
    run_tests()

