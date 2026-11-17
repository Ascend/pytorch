"""Operator lowering regressions for the DVM backend."""

import os
import unittest
from unittest import mock

import torch
import torch_npu  # noqa: F401
from torch._inductor import config


class TestDvmLowering(unittest.TestCase):
    @mock.patch.dict(os.environ, TORCHINDUCTOR_NPU_BACKEND="dvm")
    def test_sym_size_with_int_metadata(self):
        """A static aten.sym_size.int query in a dynamic graph can have int metadata."""
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

        def fn(x):
            return x + 1

        def insert_static_size_query(graph):
            tensor = next(
                n for n in graph.nodes if n.op == "placeholder"
                and isinstance(n.meta.get("val"), torch.Tensor)
            )
            output = next(n for n in graph.nodes if n.op == "output")
            value = output.args[0][0]
            # A real model hit the SymInt-only assertion on this static size query.
            # A minimal source-level repro is elusive: tracing normally folds it.
            # Insert it in a post-grad pass so int, rather than SymInt, metadata
            # reaches the actual lowering; using it in the output keeps it live.
            with graph.inserting_before(output):
                size = graph.call_function(torch.ops.aten.sym_size.int, (tensor, 1))
                size.meta["val"] = tensor.meta["val"].shape[1]
                self.assertIs(type(size.meta["val"]), int)
                scaled = graph.call_function(torch.ops.aten.mul.Scalar, (value, size))
                scaled.meta["val"] = value.meta["val"]
            output.args = ((scaled,),)

        with config.patch(fx_graph_cache=False, post_grad_custom_post_pass=insert_static_size_query):
            compiled = torch.compile(fn, dynamic=True, fullgraph=True)
            for batch in (3, 5):
                x = torch.randn(batch, 8, device="npu")
                torch._dynamo.mark_static(x, 1)
                torch.testing.assert_close(compiled(x), fn(x) * x.shape[1])

    @mock.patch.dict(os.environ, TORCHINDUCTOR_NPU_BACKEND="dvm")
    def test_sym_stride_with_int_metadata(self):
        """A dynamic tensor can have a static stride and plain int metadata."""
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

        def fn(x):
            return x + 1

        def insert_static_stride_query(graph):
            tensor = next(
                n for n in graph.nodes if n.op == "placeholder"
                and isinstance(n.meta.get("val"), torch.Tensor)
            )
            output = next(n for n in graph.nodes if n.op == "output")
            value = output.args[0][0]
            # With static width 8, dim 0 has stride 8 even for a symbolic batch.
            # Insert after folding to exercise int metadata in the actual lowering.
            with graph.inserting_before(output):
                stride = graph.call_function(torch.ops.aten.sym_stride.int, (tensor, 0))
                stride.meta["val"] = tensor.meta["val"].stride(0)
                self.assertIs(type(stride.meta["val"]), int)
                scaled = graph.call_function(torch.ops.aten.mul.Scalar, (value, stride))
                scaled.meta["val"] = value.meta["val"]
            output.args = ((scaled,),)

        with config.patch(fx_graph_cache=False, post_grad_custom_post_pass=insert_static_stride_query):
            compiled = torch.compile(fn, dynamic=True, fullgraph=True)
            for batch in (3, 5):
                x = torch.randn(batch, 8, device="npu")
                torch._dynamo.mark_static(x, 1)
                torch.testing.assert_close(compiled(x), fn(x) * x.stride(0))


if __name__ == "__main__":
    unittest.main()
