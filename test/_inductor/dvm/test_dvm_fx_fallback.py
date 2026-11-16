"""Compiled FX fallback preserves dynamic layouts and output writeback."""

import unittest
from unittest import mock

import torch
import torch_npu  # noqa: F401


class TestDvmFxFallback(unittest.TestCase):
    def test_dynamic_multioutput_and_noncontiguous_writeback(self):
        from torch._inductor import config
        from torch_npu._inductor.dvm import mlir_fusion
        from torch_npu._inductor.dvm.fx_fallback import FxGraphFallback

        def fn(x, y, out):
            value = x + y
            out.copy_(value.sum(dim=-1))
            return value.t(), out

        original_rule = mlir_fusion._is_node_supported_by_dvm_rule

        def support_rule(node, allow_common_rule=False):
            # Keep partitioning intact and force complete FX fallback at codegen.
            return original_rule(node, allow_common_rule) and allow_common_rule

        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        with (
            config.patch(fx_graph_cache=False),
            mock.patch.object(mlir_fusion, "_is_node_supported_by_dvm_rule", support_rule),
            mock.patch.object(
                FxGraphFallback, "run", autospec=True, side_effect=FxGraphFallback.run
            ) as run,
        ):
            compiled = torch.compile(
                fn, dynamic=True, fullgraph=True, options={"npu_backend": "dvm"}
            )
            for width in (7, 11, 7):
                x, y = (torch.randn(width, 5, device="npu").t() for _ in range(2))
                backing = torch.empty(5, 2, device="npu")
                out = backing[:, 0]
                stride = out.stride()
                expected = fn(x, y, torch.empty_like(out))
                run.reset_mock()
                torch.testing.assert_close(compiled(x, y, out), expected)
                torch.testing.assert_close(backing[:, 0], expected[1])
                self.assertEqual(out.stride(), stride)
                run.assert_called()
        torch.npu.synchronize()


if __name__ == "__main__":
    unittest.main()
