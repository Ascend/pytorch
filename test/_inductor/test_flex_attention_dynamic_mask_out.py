import math
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOWERING_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flex_attention.py"
TEMPLATE_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flexattention_template.py"
IR_PATH = REPO_ROOT / "torch_npu/_inductor/ir.py"
INDUCTOR_INIT_PATH = REPO_ROOT / "torch_npu/_inductor/__init__.py"
METADATA_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flex_attention_metadata.py"


try:
    import torch
    import torch_npu  # noqa: F401
    import torch_npu._inductor  # noqa: F401

    HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except (ImportError, RuntimeError):
    HAS_NPU = False


class TestFlexAttentionDynamicMaskOutSource(unittest.TestCase):
    def test_short_query_uses_community_flex_decoding(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        self.assertNotIn("upstream_flex_decoding", lowering)
        self.assertIn("def _use_flex_decoding(", lowering)
        self.assertIn("def _create_npu_flex_decoding_kernel(*args):", lowering)
        self.assertIn("flex_decoding_npu.maybe_append_choice(", lowering)
        self.assertIn('"flex_decoding",', lowering)
        self.assertIn("V.graph.sizevars.size_hint(", lowering)
        self.assertIn("config.unbacked_symint_fallback", lowering)
        self.assertNotIn("V.graph.sizevars.optimization_hint(", lowering)
        self.assertIn("V.graph.sizevars.check_leq(", lowering)
        self.assertNotIn("V.graph.sizevars.guard_leq(", lowering)
        self.assertIn(
            'kernel_options.setdefault("FLOAT32_PRECISION", "\'ieee\'")',
            lowering,
        )
        self.assertIn('cur_kernel_options.setdefault("USE_TMA", False)', lowering)

    def test_npu_template_contains_only_required_memory_changes(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        decoding = template.split("flex_decoding_npu_source =", 1)[1].split(
            "flex_decoding_npu =", 1
        )[0]
        self.assertIn("q_rows = tl.arange(0, BLOCK_M)", decoding)
        self.assertIn("q_group = q_rows // BLOCK_M_PER_HQ", decoding)
        self.assertIn("q_m = q_rows % BLOCK_M_PER_HQ", decoding)
        self.assertNotIn("q = tl.reshape(q,", decoding)
        self.assertNotIn("M_block_ptr = tl.make_block_ptr(", decoding)
        self.assertNotIn("L_block_ptr = tl.make_block_ptr(", decoding)
        self.assertIn("tl.store(M + m_offset + m_offsets", decoding)
        self.assertIn("tl.store(L + l_offset + l_offsets", decoding)

    @unittest.skip("temporarily skipped for community test")
    def test_decoding_dispatch_precedes_mask_out_dispatch(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        forward = lowering.split(
            "def _register_npu_inductor_flex_attention():", 1
        )[1]
        self.assertLess(
            forward.index("_use_flex_decoding("),
            forward.index("configured_mask_out = bool("),
        )

    def test_forced_decoding_rejects_unsupported_inputs(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        self.assertIn('backend = kernel_options.get("BACKEND", "AUTO")', lowering)
        self.assertIn(
            'if backend == "TRITON_DECODE" and not use_flex_decoding:', lowering
        )
        self.assertIn(
            "BACKEND='TRITON_DECODE' was specified but flex_decoding cannot be used",
            lowering,
        )

    def test_dynamic_paths_avoid_static_metadata_guards(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        backward = lowering.split("def flex_attention_backward(", 1)[1]
        self.assertIn("bwd_has_dynamic_shape", backward)
        self.assertIn("or bwd_has_dynamic_shape", backward)
        self.assertNotIn("NUM_SPARSE_Q_BLOCKS", lowering + template)
        self.assertNotIn("SPARSE_MASK_HQ", lowering + template)
        self.assertIn("NUM_Q_TILES = tl.cdiv(Q_LEN, BLOCK_M)", template)

    def test_pytorch_210_uses_native_subgraph_symbol_tracking(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        ir_source = IR_PATH.read_text(encoding="utf-8")
        init_source = INDUCTOR_INIT_PATH.read_text(encoding="utf-8")

        self.assertNotIn(
            "patch_triton_template_buffer_subgraph_symbols",
            ir_source + init_source,
        )
        self.assertFalse(METADATA_PATH.exists())
        self.assertIn("choices: Sequence[Any]", lowering)
        self.assertIn("expected_names", lowering)
        self.assertEqual(
            lowering.count("_filter_autotune_ir_nodes("),
            lowering.count("autotune_select_algorithm(") + 1,
        )


@unittest.skipUnless(HAS_NPU, "requires a built torch_npu package and NPU device")
class TestFlexAttentionDynamicMaskOutNPU(unittest.TestCase):
    @staticmethod
    def _causal_mask(_b, _h, q_idx, kv_idx):
        return q_idx >= kv_idx

    @staticmethod
    def _dense_reference(q, k, v, *, causal):
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        scores = scores / math.sqrt(q.size(-1))
        if causal:
            q_idx = torch.arange(q.size(-2), device=q.device)[:, None]
            kv_idx = torch.arange(k.size(-2), device=k.device)[None, :]
            scores = scores.masked_fill(q_idx < kv_idx, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, v.float()).to(q.dtype)

    def test_forward_reuses_graph_for_dynamic_block_mask_shapes(self):
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        counter = CompileCounterWithBackend("inductor")

        def fn(q, k, v, block_mask):
            return flex_attention(q, k, v, block_mask=block_mask)

        compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
        for batch, q_len, kv_len in [(5, 257, 385), (6, 129, 257)]:
            q = torch.randn(batch, 2, q_len, 64, device="npu", dtype=torch.bfloat16)
            k = torch.randn(
                batch, 2, kv_len, 64, device="npu", dtype=torch.bfloat16
            )
            v = torch.randn_like(k)
            block_mask = create_block_mask(
                self._causal_mask,
                B=batch,
                H=2,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device="npu",
            )
            self.assertFalse(
                hasattr(block_mask, "_npu_flex_attention_kernel_options")
            )
            expected = self._dense_reference(q, k, v, causal=True)
            actual = compiled(q, k, v, block_mask)
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

        self.assertEqual(counter.frame_count, 1)

    def test_forward_when_mask_mod_captures_dynamic_shape(self):
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        q_len = 128
        kv_len = 128
        q = torch.randn(
            1, 1, q_len, 64, device="npu", dtype=torch.bfloat16
        )
        k = torch.randn(
            1, 1, kv_len, 64, device="npu", dtype=torch.bfloat16
        )
        v = torch.randn_like(k)

        def window_mask(_b, _h, q_idx, kv_idx):
            return (q_idx - kv_idx).abs() <= window_source.shape[0]

        def fn(q, k, v, block_mask, window_source):
            return flex_attention(q, k, v, block_mask=block_mask)

        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(
            fn, backend=counter, dynamic=True, fullgraph=True
        )
        for window_size in (32, 48):
            window_source = torch.randn(window_size, device="npu")
            torch._dynamo.mark_dynamic(window_source, 0)
            block_mask = create_block_mask(
                window_mask,
                B=1,
                H=1,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device="npu",
            )
            self.assertFalse(
                hasattr(block_mask, "_npu_flex_attention_kernel_options")
            )
            actual = compiled(q, k, v, block_mask, window_source)

            scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
            scores = scores / math.sqrt(q.size(-1))
            q_idx = torch.arange(q_len, device="npu")[:, None]
            kv_idx = torch.arange(kv_len, device="npu")[None, :]
            outside_window = (q_idx - kv_idx).abs() > window_source.shape[0]
            scores = scores.masked_fill(outside_window, float("-inf"))
            expected = torch.matmul(torch.softmax(scores, dim=-1), v.float())
            torch.testing.assert_close(
                actual,
                expected.to(q.dtype),
                atol=2e-2,
                rtol=2e-2,
            )

        self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
    unittest.main()
