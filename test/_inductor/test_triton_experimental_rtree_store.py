# Owner(s): ["module: tests"]
# Regression test for the stale-rank store broadcast on promoted r-trees.
#
# A whole-tensor reduction whose output is a single scalar (xnumel == 1) is
# stored via an integer-index store. Upstream store codegen appends
# ``.broadcast_to(<value.shape>)`` using the CSE shape captured BEFORE the
# triton_experimental r-tree promotion raises the value's rank, so the store
# line keeps its pre-promotion rank and Triton rejects the kernel with
# ``ValueError('Cannot broadcast, rank mismatch')`` unless
# _rewrite_reduction_store_shape rebuilds it. This exercises the CausalLM loss
# pattern (shift + log_softmax + nll_loss with ignore_index) that produced
# triton_unk_fused_clone_nll_loss_forward_slice_view_* on Electra/Roberta/XGLM.

import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils

import torch_npu  # noqa: F401

# Heuristics import emitted only by the triton_experimental wrapper header
# (torch_npu/_inductor/triton_experimental/codegen/wrapper.py): identifies
# which codegen backend produced the wrapper.
EXPERIMENTAL_MARKER = "triton_experimental import npu_triton_heuristics"


class TestPromotedRtreeScalarStore(TestUtils):

    def setUp(self):
        super().setUp()
        # Reset dynamo/inductor caches so the test forces fresh codegen.
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    @staticmethod
    def _causal_lm_loss(logits, labels):
        # Shift like a CausalLM: predict labels[:, 1:] from logits[:, :-1].
        # labels[:, 1:] is a strided [B, S-1] slice and its reshape inserts a
        # clone into the graph, so the reduction index decomposes into two
        # r-nodes and the r-tree gets promoted; the loss is a single scalar
        # stored at a constant index.
        lsm = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        return F.nll_loss(
            lsm.reshape(-1, lsm.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    @parametrize("dtype", ["float32"])
    def test_promoted_rtree_scalar_store(self, dtype):
        batch, seq, vocab = 32, 512, 32
        logits = torch.randn(
            batch, seq, vocab, dtype=eval(f"torch.{dtype}"), device="npu"
        )
        labels = torch.randint(
            0, vocab, (batch, seq), dtype=torch.int64, device="npu"
        )
        labels[:, :100] = -100  # exercise the ignore_index masking path

        eager_out = self._causal_lm_loss(logits, labels)

        compiled = torch.compile(
            self._causal_lm_loss, options={"npu_backend": "triton_experimental"}
        )
        compiled_out, codes = run_and_get_code(compiled, logits, labels)

        # The compile must succeed (pre-fix it dies with the rank mismatch)
        # and produce the triton_experimental wrapper.
        self.assertIn(EXPERIMENTAL_MARKER, codes[0])
        torch.testing.assert_close(eager_out, compiled_out, rtol=1e-4, atol=1e-4)


instantiate_parametrized_tests(TestPromotedRtreeScalarStore)

if __name__ == "__main__":
    run_tests()
