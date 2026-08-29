import math
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOWERING_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flex_attention.py"
METADATA_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flex_attention_metadata.py"
TEMPLATE_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flexattention_template.py"


try:
    import torch
    import torch_npu
    # Import for torch.compile backend registration.
    import torch_npu._inductor  # noqa: F401

    HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except (ImportError, RuntimeError):
    HAS_NPU = False


class TestFlexAttentionDynamicMaskMetadataSource(unittest.TestCase):
    def test_lowering_preserves_subgraph_symbols_without_eager_metadata(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")

        self.assertIn("_filter_autotune_ir_nodes", lowering)
        self.assertIn("_attach_flex_subgraph_dependencies", lowering)
        self.assertIn("template_buffer.subgraph_inps", lowering)
        self.assertIn("template_buffer.subgraph_outs", lowering)
        self.assertNotIn("apply_kernel_options_from_eager_block_mask", lowering)
        self.assertFalse(METADATA_PATH.exists())

    def test_dynamic_forward_grid_uses_symbolic_min(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        grid_start = template.index("def flex_attention_in_loop_grid(")
        grid_end = template.index("\n\n# These metadata kernels", grid_start)
        grid_source = template[grid_start:grid_end]
        self.assertIn("min", grid_source.split("):", 1)[0])
        self.assertIn("min(total_tiles, meta[\"NUM_CUBE_CORE\"])", grid_source)


@unittest.skipUnless(HAS_NPU, "requires a built torch_npu package and NPU device")
class TestFlexAttentionDynamicMaskMetadataNPU(unittest.TestCase):
    def test_forward_when_mask_mod_captures_dynamic_shape(self):
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        q_len = 128
        kv_len = 128
        q = torch.randn(1, 1, q_len, 64, device="npu", dtype=torch.bfloat16)
        k = torch.randn(1, 1, kv_len, 64, device="npu", dtype=torch.bfloat16)
        v = torch.randn_like(k)

        def window_mask(_b, _h, q_idx, kv_idx):
            return (q_idx - kv_idx).abs() <= window_source.shape[0]

        def fn(q, k, v, block_mask, window_source):
            return flex_attention(q, k, v, block_mask=block_mask)

        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
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
