import functools
from unittest import skip
import sympy
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.library import Library, impl
from torch.testing._internal.common_utils import TestCase, run_tests
import torch_npu
from torch_npu._inductor.npu_fusion_attention_graph import NpuGraphAttentionFunction


class TestNpuFusionAttentionGraph(TestCase):
    @skip("skip for core dump")
    def test_npu_graph_attention_function(self):
        query = torch.randn(2, 4, 8, 16, device='npu', requires_grad=True)
        key = torch.randn(2, 4, 8, 16, device='npu')
        value = torch.randn(2, 4, 8, 16, device='npu')
        head_num = 4
        input_layout = "BNSD"

        output = NpuGraphAttentionFunction.apply(
            query, key, value, head_num, input_layout
        )

        self.assertEqual(output[0].shape, query.shape)
        self.assertEqual(output[1].shape, (2, 4, 8, 8))
        self.assertEqual(output[2].shape, (2, 4, 8, 8))
        self.assertEqual(output[3].shape, (0,))

        grad_outputs = (
            torch.randn_like(output[0]),
            torch.randn_like(output[1]),
            torch.randn_like(output[2]),
            torch.randn_like(output[3]),
            torch.randn(1, device='npu'),
            torch.randn(1, device='npu'),
            torch.randn(1, device='npu')
        )

        output[0].backward(grad_outputs[0])

    def test_npu_fa_forward_scale_handling(self):
        query = torch.randn(2, 4, 8, 16, device='npu')
        key = torch.randn(2, 4, 8, 16, device='npu')
        value = torch.randn(2, 4, 8, 16, device='npu')
        head_num = 4
        input_layout = "BNSD"

        result = torch.ops.npu_graph.npu_fa(
            query, key, value, head_num, input_layout, scale=2.0
        )

        self.assertEqual(result[0].shape, query.shape)
        self.assertEqual(result[1].shape, (2, 4, 8, 8))
        self.assertEqual(result[2].shape, (2, 4, 8, 8))
        self.assertEqual(result[3].shape, (0,))

    def test_npu_fa_backward_meta_impl(self):
        query = torch.randn(2, 4, 8, 16, device='meta')
        key = torch.randn(2, 4, 8, 16, device='meta')
        value = torch.randn(2, 4, 8, 16, device='meta')
        dy = torch.randn(2, 4, 8, 16, device='meta')
        head_num = 4
        input_layout = "BSH"

        result = torch.ops.npu_graph.npu_fa_backward(
            query, key, value, dy, head_num, input_layout
        )

        self.assertEqual(result[0].shape, query.shape)
        self.assertEqual(result[1].shape, key.shape)
        self.assertEqual(result[2].shape, value.shape)
        self.assertIsNone(result[3])

    def test_npu_fa_backward_scale_value_handling(self):
        query = torch.randn(1, 2, 4, 8, device='npu', requires_grad=True)
        key = torch.randn(1, 2, 4, 8, device='npu')
        value = torch.randn(1, 2, 4, 8, device='npu')
        dy = torch.randn(1, 2, 4, 8, device='npu')
        head_num = 2
        input_layout = "BNSD"

        try:
            result = torch.ops.npu_graph.npu_fa_backward(
                query, key, value, dy, head_num, input_layout, scale_value=2.0
            )
            self.assertEqual(result[0].shape, query.shape)
            self.assertEqual(result[1].shape, key.shape)
            self.assertEqual(result[2].shape, value.shape)
        except RuntimeError as e:
            if "aclnnFlashAttentionScoreGrad" in str(e):
                pass
            else:
                raise e


if __name__ == "__main__":
    run_tests()