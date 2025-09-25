import os
import unittest
from itertools import chain

import torch

os.environ["ASCEND_LAUNCH_BLOCKING"] = "0"
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestAclgraphUpdate(TestCase):

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
            softmax_lse = torch.empty(0, dtype=torch.float16, device="npu")
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
            softmax_lse = torch.empty(0, dtype=torch.float16, device="npu")
            torch_npu.npu_fused_infer_attention_score.out(
                query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, workspace=workspace,
                next_tokens=65535, softmax_lse_flag=False, actual_seq_lengths=length, out=[output, softmax_lse])
        
        g.update(cpu_update_input=[{"actual_seq_lengths": length_new}])
        g.replay()

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
            softmax_lse = torch.empty(0, dtype=torch.float16, device="npu")
            output, softmax_lse = torch_npu.npu_fused_infer_attention_score(
                query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535,
                next_tokens=65535, softmax_lse_flag=False, actual_seq_lengths=length)
        
        g.update(cpu_update_input=[{"actual_seq_lengths": length_new}])
        g.replay()
    
    @SupportedDevices(['Ascend910B'])
    @unittest.skip("test failed because of no updated CANN lib")
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
            softmax_lse = torch.empty(0, dtype=torch.float16, device="npu")
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

if __name__ == "__main__":
    run_tests()
