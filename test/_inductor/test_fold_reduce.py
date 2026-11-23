import unittest

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests

from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    fold_reduce,
)
from version_mark import runIfVersion


class TestFoldReduce(unittest.TestCase):
    @staticmethod
    def _sum_count(graph):
        return sum(
            node.target is torch.ops.aten.sum.dim_IntList
            for node in graph.nodes
        )

    @staticmethod
    def _trace_singleton_sum(x, dtype=None):
        return make_fx(
            lambda t: torch.ops.aten.sum.dim_IntList(t, [1], False, dtype=dtype),
            tracing_mode="fake",
        )(x)

    @runIfVersion(max="2.13")
    def test_dtype_converting_singleton_sum_is_not_folded(self):
        x = torch.tensor([[[True, False]], [[False, False]]])
        for dtype in (None, torch.float32):
            with self.subTest(dtype=dtype):
                graph_module = self._trace_singleton_sum(x, dtype)
                expected = graph_module(x)

                fold_reduce(graph_module.graph)

                self.assertEqual(self._sum_count(graph_module.graph), 1)
                graph_module.graph.lint()
                graph_module.recompile()
                actual = graph_module(x)
                self.assertEqual(actual.dtype, expected.dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @runIfVersion(max="2.13")
    def test_same_dtype_singleton_sum_is_folded(self):
        x = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)
        graph_module = self._trace_singleton_sum(x)
        expected = graph_module(x)

        fold_reduce(graph_module.graph)

        self.assertEqual(self._sum_count(graph_module.graph), 0)
        graph_module.graph.lint()
        graph_module.recompile()
        torch.testing.assert_close(graph_module(x), expected, rtol=0, atol=0)

    @runIfVersion(max="2.13")
    def test_missing_dtype_metadata_is_not_folded(self):
        x = torch.randn(2, 1, 3)
        for missing_dtype_node in ("input", "output"):
            with self.subTest(missing_dtype_node=missing_dtype_node):
                graph_module = self._trace_singleton_sum(x)
                sum_node = next(
                    node
                    for node in graph_module.graph.nodes
                    if node.target is torch.ops.aten.sum.dim_IntList
                )
                node = sum_node.args[0] if missing_dtype_node == "input" else sum_node
                for key in ("val", "tensor_meta", "dtype"):
                    node.meta.pop(key, None)

                fold_reduce(graph_module.graph)

                self.assertEqual(self._sum_count(graph_module.graph), 1)


if __name__ == "__main__":
    run_tests()
