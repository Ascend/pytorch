from dataclasses import dataclass

import random
import numpy as np
import torch

import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


@dataclass
class PAAttentionParamsNumpy:
    query: np.ndarray
    key_cache: np.ndarray
    value_cache: np.ndarray
    block_table: np.ndarray
    context_lens: np.ndarray


@dataclass
class PAAttentionParamsTensor:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor


class TestPAAclgraphUpdate(TestCase):
    num_blocks = 64
    num_tokens = 2
    block_size = 128
    kv_heads = 16
    head_size = 288
    num_heads = 32
    head_size_v = 96
    scale = 0.38888

    def group_matmul(self, head, kv_head, A, B):
        group_num = head // kv_head
        score = []
        for i in range(kv_head):
            group_A = A[i * group_num: (i + 1) * group_num]
            group_B = B[i: i + 1]
            score.append(np.matmul(group_A, group_B))
        return np.concatenate(score, axis=0)

    def ref_masked_attention(self, query, key, value):
        """参考注意力计算"""
        # 维度调整 [num_heads, seq_len, head_size]
        query = query * self.scale
        query = query.transpose(1, 0, 2)
        key = key.transpose(1, 2, 0)

        # QK^T计算
        sim = self.group_matmul(query.shape[0], key.shape[0], query, key)

        # Softmax归一化
        sim = sim - np.max(sim, axis=-1, keepdims=True)
        exp_sim = np.exp(sim.astype(np.float32))
        p = exp_sim / np.sum(exp_sim, axis=-1, keepdims=True)
        p = p.astype(np.float16)

        # Value加权
        value = value.transpose(1, 0, 2)
        out = self.group_matmul(p.shape[0], key.shape[0], p, value)
        return out.transpose(1, 0, 2)

    def golden_attention_impl(self, params_np):
        output = np.zeros((self.num_tokens, self.num_heads, self.head_size_v), dtype=np.float16)

        for i in range(self.num_tokens):
            # 从缓存中收集当前序列的KV
            seq_blocks = params_np.block_table[i]
            context_len = params_np.context_lens[i]

            keys = []
            values = []
            for pos in range(context_len):
                block_id = seq_blocks[pos // self.block_size]
                offset = pos % self.block_size
                keys.append(params_np.key_cache[block_id, offset].reshape(self.kv_heads, -1))
                values.append(params_np.value_cache[block_id, offset].reshape(self.kv_heads, -1))

            # 执行注意力计算
            out = self.ref_masked_attention(
                params_np.query[i:i + 1], 
                np.stack(keys), 
                np.stack(values)
            )
            output[i] = out.reshape(self.num_heads, -1)
        return output

    def preprocess(self):
        """生成测试输入数据"""
        query_np = np.random.uniform(-1, 1, (self.num_tokens, self.num_heads, self.head_size)).astype(np.float16)
        key_cache_np = np.random.uniform(-1, 1, (self.num_blocks, self.block_size, self.kv_heads, self.head_size)).astype(np.float16)
        value_cache_np = np.random.uniform(-1, 1, (self.num_blocks, self.block_size, self.kv_heads, self.head_size_v)).astype(np.float16)
        max_blocks_per_seq = (1024 + self.block_size - 1) // self.block_size
        block_table_np = np.array([
            [random.randint(0, self.num_blocks - 1) for _ in range(max_blocks_per_seq)]
            for _ in range(self.num_tokens)
        ], dtype=np.int32)
        context_lens_np = np.full(self.num_tokens, random.randint(1, 1024), dtype=np.int32)
        params_np = PAAttentionParamsNumpy(query_np, key_cache_np, value_cache_np, block_table_np, context_lens_np)
        golden_output = self.golden_attention_impl(params_np)
        golden_output = torch.from_numpy(golden_output)

        query = torch.from_numpy(query_np).npu()
        key_cache = torch.from_numpy(key_cache_np).npu()
        value_cache = torch.from_numpy(value_cache_np).npu()
        block_table = torch.from_numpy(block_table_np).npu()
        self.context_lens = torch.from_numpy(context_lens_np)
        self.output = torch.zeros_like(query[:, :, :self.head_size_v]).npu()
        params_tensor = PAAttentionParamsTensor(query, key_cache, value_cache, block_table)
        return params_tensor, golden_output

    def atb_paged_attention(self, query, key_cache, value_cache, block_table):
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=self.kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=block_table,
            context_lens=self.context_lens,
            out=self.output,
        )

    @SupportedDevices(['Ascend910B'])
    def test_paged_attention_compile_with_aclgraph(self):
        params, golden_output = self.preprocess()

        compiled_fn = torch.compile(self.atb_paged_attention, backend="inductor", dynamic=False, options={"triton.cudagraphs": True})
        compiled_fn(params.query, params.key_cache, params.value_cache, params.block_table)
        self.assertRtolEqual(self.output, golden_output, prec16=0.01)

        params, golden_output = self.preprocess()
        compiled_fn(params.query, params.key_cache, params.value_cache, params.block_table)
        self.assertRtolEqual(self.output, golden_output, prec16=0.01)

if __name__ == "__main__":
    run_tests()