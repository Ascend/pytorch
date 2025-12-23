import unittest
from dataclasses import dataclass
from itertools import chain
import os

import random
import numpy as np
import torch

import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestIFAAclgraphUpdate(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_ifa_update(self):
        torch.npu.set_device(0)
        length = [29]
        length_new = [100]
        scale = 1 / 0.0078125
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")
        key = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")
        value = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")

        res_src = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535,
            next_tokens=65535, softmax_lse_flag=False, actual_seq_lengths=length_new)

        g = torch.npu.NPUGraph()
        event = torch.npu.ExternalEvent()
        update_stream = torch.npu.Stream()
        handle = None
        output = None
        softmax_lse = None

        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535,
            next_tokens=65535, softmax_lse_flag=False, actual_seq_lengths=length)

        with torch.npu.graph(g):
            stream = torch.npu.current_stream()
            output = torch.empty(1, 32, 1, 128, dtype=torch.float16, device="npu")
            softmax_lse = torch.empty_like(res_src[1], dtype=torch.float16, device="npu")
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, workspace=workspace,
                next_tokens=65535, softmax_lse_flag=False, actual_seq_lengths=length, out=[output, softmax_lse])
            handle = torch.npu.graph_task_group_end(stream)
        
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(
                query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, workspace=workspace,
                next_tokens=65535, softmax_lse_flag=False, actual_seq_lengths=length_new, out=[output, softmax_lse])
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)

        g.replay()
        self.assertEqual(output.cpu(), res_src[0].cpu())
        self.assertEqual(softmax_lse.cpu(), res_src[1].cpu())

    @SupportedDevices(['Ascend910B'])
    def test_update_stream_globally_unique(self):
        torch.npu.set_device(0)

        g1 = torch.npu.NPUGraph()
        g2 = torch.npu.NPUGraph()
        self.assertEqual(g1.graph_dispatch_mode.update_stream, g2.graph_dispatch_mode.update_stream)

    @SupportedDevices(['Ascend910B'])
    def test_ifa_update_with_auto_dispatch_capture(self):
        torch.npu.set_device(0)
        length = [29]
        length_new = [100]
        scale = 1 / 0.0078125
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")
        key = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")
        value = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")

        res_src = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535,
            next_tokens=65535, softmax_lse_flag=False, actual_seq_lengths=length_new)

        g = torch.npu.NPUGraph()
        output = None
        softmax_lse = None

        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535,
            next_tokens=65535, softmax_lse_flag=False, actual_seq_lengths=length)

        with torch.npu.graph(g, auto_dispatch_capture=True):
            output = torch.empty(1, 32, 1, 128, dtype=torch.float16, device="npu")
            softmax_lse = torch.empty_like(res_src[1], dtype=torch.float16, device="npu")
            torch_npu.npu_fused_infer_attention_score.out(
                query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, workspace=workspace,
                next_tokens=65535, softmax_lse_flag=False, actual_seq_lengths=length, out=[output, softmax_lse])
        
        g.update(cpu_update_input=[{"actual_seq_lengths": length_new}])
        g.replay()
        self.assertEqual(output.cpu(), res_src[0].cpu())
        self.assertEqual(softmax_lse.cpu(), res_src[1].cpu())

    @SupportedDevices(['Ascend910B'])
    def test_ifa_update_with_non_out_and_auto_dispatch_capture(self):
        torch.npu.set_device(0)
        length = [29]
        length_new = [100]
        scale = 1 / 0.0078125
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")
        key = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")
        value = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")

        res_src = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535,
            next_tokens=65535, softmax_lse_flag=False, actual_seq_lengths=length_new)

        g = torch.npu.NPUGraph()
        output = None
        softmax_lse = None

        with torch.npu.graph(g, auto_dispatch_capture=True):
            output = torch.empty(1, 32, 1, 128, dtype=torch.float16, device="npu")
            softmax_lse = torch.empty(1, dtype=torch.float16, device="npu")
            output, softmax_lse = torch_npu.npu_fused_infer_attention_score(
                query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535,
                next_tokens=65535, softmax_lse_flag=False, actual_seq_lengths=length)
        
        g.update(cpu_update_input=[{"actual_seq_lengths": length_new}])
        g.replay()
        self.assertEqual(output.cpu(), res_src[0].cpu())
        self.assertEqual(softmax_lse.cpu(), res_src[1].cpu())
    
    @SupportedDevices(['Ascend910B'])
    @unittest.skip("this cann version is not supported")
    def test_npu_fused_infer_attention_score_v2(self):
        torch.npu.set_device(0)
        length = [29]
        length_new = [100]
        scale = 1 / 0.0078125
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")
        key = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")
        value = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")
        res_src = torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=32, input_layout="BNSD", softmax_scale=scale, pre_tokens=65535,
            next_tokens=65535, return_softmax_lse=False, actual_seq_qlen=length_new)
        g = torch.npu.NPUGraph()
        event = torch.npu.ExternalEvent()
        update_stream = torch.npu.Stream()
        handle = None
        output = None
        softmax_lse = None

        workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
            query, key, value, num_query_heads=32, input_layout="BNSD", softmax_scale=scale, pre_tokens=65535,
            next_tokens=65535, return_softmax_lse=False, actual_seq_qlen=length)

        with torch.npu.graph(g):
            stream = torch.npu.current_stream()
            output = torch.empty(1, 32, 1, 128, dtype=torch.float16, device="npu")
            softmax_lse = torch.empty(1, dtype=torch.float16, device="npu")
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score_v2.out(
                query, key, value, num_query_heads=32, input_layout="BNSD", softmax_scale=scale, pre_tokens=65535, workspace=workspace,
                next_tokens=65535, return_softmax_lse=False, actual_seq_qlen=length, out=[output, softmax_lse])
            handle = torch.npu.graph_task_group_end(stream)
        
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score_v2.out(
                query, key, value, num_query_heads=32, input_layout="BNSD", softmax_scale=scale, pre_tokens=65535, workspace=workspace,
                next_tokens=65535, return_softmax_lse=False, actual_seq_qlen=length_new, out=[output, softmax_lse])
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)

        g.replay()
        self.assertEqual(output.cpu(), res_src[0].cpu())
        self.assertEqual(softmax_lse.cpu(), res_src[1].cpu())

    @SupportedDevices(['Ascend910B'])
    @unittest.skip("this cann version is not supported")
    def test_npugraph_debug_dump(self):
        N, D_in, H, D_out = 640, 4096, 2048, 1024
        model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                                    torch.nn.Dropout(p=0.2),
                                    torch.nn.Linear(H, D_out),
                                    torch.nn.Dropout(p=0.1)).npu()
        
        static_input = torch.randn(N, D_in, device='npu')
        s = torch.npu.Stream()
        s.wait_stream(torch.npu.current_stream())
        model.eval()
        with torch.npu.stream(s):
            for _ in range(3):
                y_pred = model(static_input)
        torch.npu.current_stream().wait_stream(s)
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            static_y_pred = model(static_input)

        file_path = os.path.join(os.getcwd(), "jsonPrint.json")
        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)

        g.debug_dump(file_path)
        self.assertTrue(os.path.getsize(file_path) > 0, "npugraph debug dump assert error")
        os.remove(file_path) 


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
    context_lens: torch.Tensor
    output: torch.Tensor


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
        context_lens = torch.from_numpy(context_lens_np)
        output = torch.zeros_like(query[:, :, :self.head_size_v]).npu()
        params_tensor = PAAttentionParamsTensor(query, key_cache, value_cache, block_table, context_lens, output)
        return params_tensor, golden_output

    def atb_paged_attention(self, params):
        torch_npu._npu_paged_attention(
            query=params.query, 
            key_cache=params.key_cache, 
            value_cache=params.value_cache,
            num_kv_heads=self.kv_heads,
            num_heads=self.num_heads, 
            scale_value=self.scale,
            block_table=params.block_table,
            context_lens=params.context_lens,
            out=params.output,
        )
        return params.output

    @SupportedDevices(['Ascend910B'])
    def test_paged_attention_aclgraph_update(self):
        params, golden_output = self.preprocess()
        output = None

        # capture
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph,
                             stream=torch.npu.Stream(),
                             pool=None,
                             auto_dispatch_capture=True):
            output = self.atb_paged_attention(params)
        graph.update(cpu_update_input=[{"context_lens": params.context_lens}])
        graph.replay()
        torch.npu.synchronize()
        self.assertRtolEqual(output, golden_output, prec16=0.01)

        params_new, golden_output = self.preprocess()
        params.query.copy_(params_new.query)
        params.key_cache.copy_(params_new.key_cache)
        params.value_cache.copy_(params_new.value_cache)
        params.block_table.copy_(params_new.block_table)
        graph.update(cpu_update_input=[{"context_lens": params_new.context_lens}])
        graph.replay()
        torch.npu.synchronize()
        self.assertRtolEqual(output, golden_output, prec16=0.01)

if __name__ == "__main__":
    run_tests()
