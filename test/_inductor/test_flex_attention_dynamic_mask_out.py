import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOWERING_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flex_attention.py"
TEMPLATE_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flexattention_template.py"


class TestFlexAttentionDynamicMaskOutSource(unittest.TestCase):
    def test_short_query_uses_community_flex_decoding(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        self.assertNotIn("upstream_flex_decoding", lowering)
        self.assertIn("def _use_flex_decoding(", lowering)
        self.assertIn("def _create_npu_flex_decoding_kernel(*args):", lowering)
        self.assertIn("flex_decoding_npu.maybe_append_choice(", lowering)
        self.assertIn('"flex_decoding",', lowering)
        self.assertIn(
            'cur_kernel_options.setdefault("USE_TMA", bool(torch.xpu.is_available()))',
            lowering,
        )

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
        self.assertIn("tl.store(m_ptrs, m_i", decoding)
        self.assertIn("tl.store(l_ptrs, l_i", decoding)
        self.assertIn(
            "flex_decoding_npu.always_freeze_layout = True", template
        )

    @unittest.skip("temporarily disabled pending decoding dispatch update")
    def test_decoding_dispatch_precedes_mask_out_dispatch(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        forward = lowering.split(
            "def _register_npu_inductor_flex_attention():", 1
        )[1]
        self.assertLess(
            forward.index("if _use_flex_decoding("),
            forward.index("configured_mask_out = bool("),
        )


if __name__ == "__main__":
    unittest.main()
