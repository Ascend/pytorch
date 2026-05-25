# Owner(s): ["module: inductor"]
"""End-to-end tests for FA v3 graph partition pass."""

import functools

import numpy as np
import torch_npu
from testutils import TestUtils
from torch_npu.testing.common_utils import SupportedDevices

import torch
from torch.testing._internal.common_utils import run_tests


# Loose tolerance for dropout paths (RNG ordering between eager and compiled
# wrapper differs slightly even when the underlying FA v3 op runs in eager).
_HEAD_NUM = 8
_HEAD_DIM = 64
_DTYPE = torch.float16


def _make_bnsd_inputs(batch=2, seq=128):
    shape = (batch, _HEAD_NUM, seq, _HEAD_DIM)
    q = torch.randn(shape, dtype=_DTYPE, device="npu", requires_grad=False)
    k = torch.randn(shape, dtype=_DTYPE, device="npu", requires_grad=False)
    v = torch.randn(shape, dtype=_DTYPE, device="npu", requires_grad=False)
    return q, k, v


def _make_tnd_inputs(batch=2, seq=128):
    total_tokens = batch * seq
    shape = (total_tokens, _HEAD_NUM, _HEAD_DIM)
    q = torch.randn(shape, dtype=_DTYPE, device="npu", requires_grad=False)
    k = torch.randn(shape, dtype=_DTYPE, device="npu", requires_grad=False)
    v = torch.randn(shape, dtype=_DTYPE, device="npu", requires_grad=False)
    # FA v3 schema requires actual_seq_qlen / actual_seq_kvlen to be Tensor (CPU int64).
    cu_seqlens = torch.tensor(
        np.arange(1, batch + 1) * seq,
        dtype=torch.int64,
    )
    return q, k, v, cu_seqlens


def _fa3_bnsd(q, k, v, keep_prob):
    out = torch.ops.npu.npu_fusion_attention_v3(
        q,
        k,
        v,
        _HEAD_NUM,
        "BNSD",
        keep_prob=keep_prob,
    )
    return out[0]


def _fa3_tnd(q, k, v, actual_seq_qlen, actual_seq_kvlen, keep_prob):
    out = torch.ops.npu.npu_fusion_attention_v3(
        q,
        k,
        v,
        _HEAD_NUM,
        "TND",
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        keep_prob=keep_prob,
    )
    return out[0]


class TestFAv3PartitionPass(TestUtils):
    """End-to-end tests for `register_fav3_partition_pass`."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Importing the inductor subpackage triggers register_fav3_partition_pass()
        # via the wiring in torch_npu/_inductor/__init__.py.
        from torch_npu._inductor.fx_passes import _proxy_ops, fav3_partition_pass

        cls._pmp = fav3_partition_pass._FAV3_PMP
        cls._proxy_targets = set(_proxy_ops.PROXY_TARGETS.values())

        # Disable inductor's persistent FX graph cache so each test triggers a
        # fresh post_grad_passes run -- otherwise the second compile of the
        # same callable hits cached output, our pass is skipped, and rewrite
        # counters falsely report 0.
        cls._fx_cache_orig = torch._inductor.config.fx_graph_cache
        torch._inductor.config.fx_graph_cache = False
        from torch_npu._inductor import config as npu_config

        npu_config.npugraph_trees.disable_cpu_input_check = True

    @classmethod
    def tearDownClass(cls):
        torch._inductor.config.fx_graph_cache = cls._fx_cache_orig
        super().tearDownClass()
        from torch_npu._inductor import config as npu_config

        npu_config.npugraph_trees.disable_cpu_input_check = False

    def setUp(self):
        super().setUp()
        # Wrap _FAV3_PMP.apply to count rewrites this test causes.
        self._rewrite_count = 0
        original_apply = type(self)._pmp.apply

        @functools.wraps(original_apply)
        def counting_apply(gm):
            graph = gm.graph if hasattr(gm, "graph") else gm
            before = sum(
                1
                for n in graph.nodes
                if n.op == "call_function" and n.target in type(self)._proxy_targets
            )
            result = original_apply(gm)
            after = sum(
                1
                for n in graph.nodes
                if n.op == "call_function" and n.target in type(self)._proxy_targets
            )
            self._rewrite_count += max(0, after - before)
            return result

        self._original_apply = original_apply
        type(self)._pmp.apply = counting_apply
        torch._dynamo.reset()

    def tearDown(self):
        type(self)._pmp.apply = self._original_apply
        torch._dynamo.reset()
        super().tearDown()

    # ---- helpers --------------------------------------------------------

    def _run_eager_then_compiled(self, fn, args, compile_mode):
        torch.manual_seed(42)
        torch_npu.npu.manual_seed(42)
        eager_out = fn(*args)

        torch.manual_seed(42)
        torch_npu.npu.manual_seed(42)
        compiled_fn = torch.compile(fn, mode=compile_mode, dynamic=False)
        compiled_out = compiled_fn(*args)
        return eager_out, compiled_out

    # ---- scenarios ------------------------------------------------------

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_tnd_dropout_reduce_overhead_partitions(self):
        """Regression target: TND + dropout must be partitioned out of ACLgraph."""
        torch.manual_seed(0)
        q, k, v, cu = _make_tnd_inputs()
        eager_out, compiled_out = self._run_eager_then_compiled(
            _fa3_tnd,
            (q, k, v, cu, cu, 0.5),
            compile_mode="reduce-overhead",
        )

        # If the partition pass didn't rewrite, ACLgraph capture would crash
        # before reaching here -- so reaching the assertion is itself half the
        # signal. We additionally require a measurable rewrite.
        self.assertGreaterEqual(
            self._rewrite_count,
            1,
            f"expected >=1 FA v3 node to be rewritten, got {self._rewrite_count}",
        )
        self.assertEqual(eager_out, compiled_out)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_tnd_no_dropout_reduce_overhead_no_rewrite(self):
        """TND with keep_prob=1.0 is safe in ACLgraph; no rewrite expected."""
        torch.manual_seed(0)
        q, k, v, cu = _make_tnd_inputs()
        eager_out, compiled_out = self._run_eager_then_compiled(
            _fa3_tnd,
            (q, k, v, cu, cu, 1.0),
            compile_mode="reduce-overhead",
        )

        self.assertEqual(
            self._rewrite_count,
            0,
            f"expected no FA v3 rewrite, got {self._rewrite_count}",
        )
        self.assertEqual(eager_out, compiled_out)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_bnsd_dropout_reduce_overhead_no_rewrite(self):
        """BNSD always safe regardless of dropout; no rewrite expected."""
        torch.manual_seed(0)
        q, k, v = _make_bnsd_inputs()
        eager_out, compiled_out = self._run_eager_then_compiled(
            _fa3_bnsd,
            (q, k, v, 0.5),
            compile_mode="reduce-overhead",
        )

        self.assertEqual(
            self._rewrite_count,
            0,
            f"expected no FA v3 rewrite for BNSD, got {self._rewrite_count}",
        )
        self.assertEqual(eager_out, compiled_out)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_tnd_dropout_default_mode_early_out(self):
        """cudagraphs=OFF early-out: pass must be a no-op even with TND+dropout."""
        torch.manual_seed(0)
        q, k, v, cu = _make_tnd_inputs()
        eager_out, compiled_out = self._run_eager_then_compiled(
            _fa3_tnd,
            (q, k, v, cu, cu, 0.5),
            compile_mode="default",
        )

        self.assertEqual(
            self._rewrite_count,
            0,
            f"cudagraphs=OFF early-out: expected no rewrite, got {self._rewrite_count}",
        )
        self.assertEqual(eager_out, compiled_out)


if __name__ == "__main__":
    run_tests()
