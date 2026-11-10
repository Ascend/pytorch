import unittest

import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests
from testutils import TestUtils

import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    singleton_index_broadcast_pass,
)


class SingletonIndexCloneModel(torch.nn.Module):
    def forward(self, table, index):
        selected = torch.ops.aten.index.Tensor(table, [index])
        return torch.ops.aten.clone.default(
            selected, memory_format=torch.contiguous_format
        )


class TestSingletonIndexBroadcastPass(TestUtils):
    @unittest.skipUnless(torch_npu.npu.is_available(), "requires an NPU")
    def test_singleton_index_clone_compile(self):
        # A non-contiguous single-row table exercises the clone-reuse path.
        table = torch.arange(24, dtype=torch.float16, device="npu")
        table = table.reshape(1, 3, 2, 4).permute(0, 2, 1, 3)
        index = torch.tensor([0, -1, 0, -1], dtype=torch.int64, device="npu")
        model = SingletonIndexCloneModel()

        with torch.no_grad():
            expected = model(table, index)
            graph_module = make_fx(model, tracing_mode="fake")(table, index)
            self.assertEqual(
                sum(n.target == torch.ops.aten.index.Tensor for n in graph_module.graph.nodes),
                1,
            )

            singleton_index_broadcast_pass(graph_module.graph)
            graph_module.recompile()

            self.assertFalse(
                any(n.target == torch.ops.aten.index.Tensor for n in graph_module.graph.nodes)
            )
            self.assertEqual(
                sum(n.target == torch.ops.aten.expand.default for n in graph_module.graph.nodes),
                1,
            )
            self.assertEqual(
                sum(n.target == torch.ops.aten.clone.default for n in graph_module.graph.nodes),
                1,
            )
            rewritten = graph_module(table, index)
            self.assertEqual(expected, rewritten, atol=0, rtol=0)
            self.assertEqual(expected.dtype, rewritten.dtype)
            self.assertEqual(expected.shape, rewritten.shape)
            self.assertEqual(expected.stride(), rewritten.stride())
            self.assertNotEqual(table.data_ptr(), rewritten.data_ptr())

            matches_before = counters["inductor"]["npu_singleton_index_broadcast"]
            compiled = torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)
            actual = compiled(table, index)
            self.assertGreater(
                counters["inductor"]["npu_singleton_index_broadcast"], matches_before
            )
            self.assertEqual(expected, actual, atol=0, rtol=0)
            self.assertEqual(expected.stride(), actual.stride())
            self.assertNotEqual(table.data_ptr(), actual.data_ptr())


if __name__ == "__main__":
    run_tests()
