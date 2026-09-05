import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOWERING_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flex_attention.py"
TEMPLATE_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flexattention_template.py"


class TestFlexAttentionDynamicMaskOutSource(unittest.TestCase):
    def test_short_query_uses_community_flex_decoding(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        self.assertIn("flex_decoding_template,", lowering)
        self.assertIn("def _use_flex_decoding(", lowering)
        self.assertIn("def _create_npu_flex_decoding_kernel(*args):", lowering)
        self.assertIn(
            "flex_decoding_template.maybe_append_choice(",
            lowering,
        )
        self.assertIn('"flex_decoding",', lowering)
        self.assertIn(
            'cur_kernel_options.setdefault("USE_TMA", bool(torch.xpu.is_available()))',
            lowering,
        )

    def test_npu_decoding_uses_community_template(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("flex_decoding_npu", template)
        self.assertNotIn("flex_decoding_npu_source", template)
        self.assertIn(
            "flex_decoding_template = _wrap_upstream_template(", template
        )
        self.assertIn("_upstream_flex_decoding_template", template)

    @unittest.skip("temporarily disabled pending decoding dispatch update")
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


if __name__ == "__main__":
    unittest.main()
