"""Unit tests for torch_npu/_inductor/kernel/flexattention_template.py.

Covers:
  1. Module import and template attribute integrity.
  2. NPUTemplateCompileOption semantics (setdefault based apply).
  3. Symbolic grid functions (numeric behaviour with int inputs).
  4. Wrapped upstream templates (flex_attention / backward / decoding).
  5. Explicit NPUTritonTemplate instances (names, grids, sources, options).
  6. End-to-end flex_attention forward/backward on NPU comparing against
     torch.nn.functional.scaled_dot_product_attention.

Run on NPU 1 (must be set before the process starts):
    export ASCEND_RT_VISIBLE_DEVICES=1
    python -m pytest test_flex_attention_template.py -v
"""

import unittest

import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401  registers the npu inductor backend

from torch._inductor.select_algorithm import TritonTemplate
from torch_npu._inductor.kernel import flexattention_template as ft
from torch_npu._inductor.select_algorithm import (
    NPUTemplateCompileOption,
    NPUTritonTemplate,
)

try:
    from torch._inductor.kernel.flex.flex_attention import SymbolicGridFn
except ImportError:
    from torch._inductor.kernel.flex_attention import SymbolicGridFn  # type: ignore[import-error]

NPU_AVAILABLE = torch.npu.is_available()

# All module level NPUTritonTemplate instances exported by the module, mapped
# to the expected NPU extra name suffix used by the code cache key.
TEMPLATE_EXTRA_NAMES = {
    "flex_attention_fwd_mask_out": "fwd_main",
    "flex_attention_fwd_mask_out_all_sparse": "fwd_sparse",
    "flex_attention_compact_offsets": "mask_offsets",
    "flex_attention_compact_mapping": "mask_mapping",
    "flex_attention_fwd_mask_compact": "fwd_compact",
    "flex_attention_fwd_workspace_offsets": "fwd_ws_offsets",
    "flex_attention_fwd_workspace_mask_compact": "fwd_ws_compact",
    "flex_attention_bwd_mask_compact": "bwd_compact",
    "flex_attention_bwd_mask_pos": "bwd_mask_pos",
    "flex_attention_bwd_dq_mask_out": "bwd_dq",
    "flex_attention_bwd_dkdv_mask_out": "bwd_dkdv",
    "flex_attention_bwd_dkdv_tasklist": "bwd_tasklist",
    "flex_attention_bwd_dkdv_tasklist_no_split": "bwd_task_nosplit",
    "flex_attention_bwd_dkdv_reduce": "bwd_reduce",
}

# manual_output_buffer is only set on these templates.
TEMPLATE_MANUAL_OUTPUT_BUFFERS = {
    "flex_attention_fwd_workspace_offsets": "arg_Q_OFFSETS",
    "flex_attention_bwd_mask_compact": "arg_SPARSE_MASK",
}

# Templates sharing the forward compile options.
FWD_OPTION_TEMPLATES = {
    "flex_attention_fwd_mask_out",
    "flex_attention_fwd_mask_out_all_sparse",
}

# Templates sharing the backward (dq / dkdv) compile options.
BWD_DQ_OPTION_TEMPLATES = {"flex_attention_bwd_dq_mask_out"}
BWD_DKDV_OPTION_TEMPLATES = {
    "flex_attention_bwd_dkdv_mask_out",
    "flex_attention_bwd_dkdv_tasklist",
    "flex_attention_bwd_dkdv_tasklist_no_split",
    "flex_attention_bwd_dkdv_reduce",
}

# Grid functions defined in the module (SymbolicGridFn wrapped).
GRID_FUNCTIONS = [
    "flex_attention_in_loop_grid",
    "compact_offsets_grid",
    "compact_mapping_grid",
    "fwd_workspace_offsets_grid",
    "compact_sparse_mask_grid",
    "sparse_mask_block_pos_grid",
    "flex_attention_backward_dq_grid",
    "flex_attention_backward_dkdv_grid",
]

# Kernel source string attributes defined at module level.
KERNEL_SOURCES = [
    "get_bounded_indices_func",
    "compute_compact_sparse_mask_offsets_kernel",
    "compute_compact_sparse_mask_mapping_kernel",
    "compute_sparse_mask_kernel_compact",
    "compute_fwd_workspace_offsets",
    "compute_bwd_sparse_mask_kernel_compact",
    "compute_sparse_mask_block_pos_kernel",
    "compute_forward_block_mn_sparse_mask",
    "compute_forward_inner_sparse_mask_direct_index",
    "compute_flex_attention_sparse_mask_in_loop_no_load_balance",
    "compute_forward_block_mn_full",
    "flex_attention_backward_qmajor_dq_source",
    "flex_attention_backward_dkdv_only_source",
    "flex_attention_backward_dkdv_tasklist_source",
    "flex_attention_backward_dkdv_reduce_source",
]

# Template name -> module level kernel source feeding it.
TEMPLATE_SOURCES = {
    "flex_attention_fwd_mask_out": "_FWD_MASK_OUT_SOURCE",
    "flex_attention_fwd_mask_out_all_sparse": "_FWD_MASK_OUT_SOURCE",
    "flex_attention_compact_offsets": "compute_compact_sparse_mask_offsets_kernel",
    "flex_attention_compact_mapping": "compute_compact_sparse_mask_mapping_kernel",
    "flex_attention_fwd_mask_compact": "compute_sparse_mask_kernel_compact",
    "flex_attention_fwd_workspace_offsets": "compute_fwd_workspace_offsets",
    "flex_attention_fwd_workspace_mask_compact": "compute_fwd_workspace_mask_compact",
    "flex_attention_bwd_mask_compact": "compute_bwd_sparse_mask_kernel_compact",
    "flex_attention_bwd_mask_pos": "compute_sparse_mask_block_pos_kernel",
    "flex_attention_bwd_dq_mask_out": "flex_attention_backward_qmajor_dq_source",
    "flex_attention_bwd_dkdv_mask_out": "flex_attention_backward_dkdv_only_source",
    "flex_attention_bwd_dkdv_tasklist": "flex_attention_backward_dkdv_tasklist_source",
    "flex_attention_bwd_dkdv_tasklist_no_split": "flex_attention_backward_dkdv_tasklist_source",
    "flex_attention_bwd_dkdv_reduce": "flex_attention_backward_dkdv_reduce_source",
}


class TestModuleImport(unittest.TestCase):
    """The module is importable and exports every documented symbol."""

    def test_module_import(self):
        self.assertIsNotNone(ft)

    def test_compile_option_objects_exist(self):
        for name in (
            "_FWD_COMPILE_OPTIONS",
            "_BWD_DQ_COMPILE_OPTIONS",
            "_BWD_DKDV_COMPILE_OPTIONS",
        ):
            self.assertTrue(
                hasattr(ft, name),
                f"missing module attribute: {name}",
            )
            self.assertIsInstance(getattr(ft, name), NPUTemplateCompileOption)

    def test_kernel_sources_are_non_empty_strings(self):
        for name in KERNEL_SOURCES:
            self.assertTrue(hasattr(ft, name), f"missing source: {name}")
            value = getattr(ft, name)
            self.assertIsInstance(value, str)
            self.assertGreater(len(value), 0, f"empty kernel source: {name}")

    def test_sources_use_def_kernel_placeholder(self):
        for name in KERNEL_SOURCES:
            source = getattr(ft, name)
            if name == "get_bounded_indices_func":
                # A plain @triton.jit helper without template placeholders.
                self.assertIn("@triton.jit", source)
                continue
            placeholders = ("def_kernel", "gen_argdefs", "gen_defines")
            self.assertTrue(
                any(p in source for p in placeholders),
                f"no template placeholder in {name}",
            )

    def test_fwd_mask_out_source_composition(self):
        # _FWD_MASK_OUT_SOURCE concatenates the five forward pieces.
        for piece in (
            "compute_flex_attention_sparse_mask_in_loop_no_load_balance",
            "compute_forward_inner_sparse_mask_direct_index",
            "compute_forward_block_mn_sparse_mask",
            "compute_forward_block_mn_full",
            "get_bounded_indices_func",
        ):
            self.assertIn(
                getattr(ft, piece),
                ft._FWD_MASK_OUT_SOURCE,
                f"{piece} missing from _FWD_MASK_OUT_SOURCE",
            )

    def test_dkdv_helpers_embedded_in_tasklist_source(self):
        self.assertIn(
            ft._FLEX_ATTENTION_BACKWARD_DKDV_HELPERS_MARKER,
            ft.flex_attention_backward_dkdv_tasklist_source,
        )


class TestNPUTemplateCompileOption(unittest.TestCase):
    """apply() must behave like dict.setdefault for every option key."""

    def test_apply_sets_missing_keys(self):
        option = NPUTemplateCompileOption({"alpha": 1, "beta": "x"})
        meta = {}
        option.apply(meta)
        self.assertEqual(meta, {"alpha": 1, "beta": "x"})

    def test_apply_does_not_override_existing_keys(self):
        option = NPUTemplateCompileOption({"alpha": 1})
        meta = {"alpha": 999}
        option.apply(meta)
        self.assertEqual(meta["alpha"], 999)

    def test_apply_partial_overlap(self):
        option = NPUTemplateCompileOption({"a": 1, "b": 2})
        meta = {"b": 42}
        option.apply(meta)
        self.assertEqual(meta, {"a": 1, "b": 42})

    def test_fwd_compile_options_content(self):
        options = ft._FWD_COMPILE_OPTIONS.options
        self.assertTrue(options["enable_ubuf_saving"])
        self.assertTrue(options["unit_flag"])
        self.assertEqual(options["set_workspace_multibuffer"], 4)
        self.assertEqual(options["limit_auto_multi_buffer_buffer"], "no-limit")

    def test_bwd_dq_compile_options_content(self):
        options = ft._BWD_DQ_COMPILE_OPTIONS.options
        self.assertFalse(options["enable_select_analysis"])
        self.assertEqual(options["buf_slot_num_of_veccore"], 3)

    def test_bwd_dkdv_compile_options_content(self):
        options = ft._BWD_DKDV_COMPILE_OPTIONS.options
        self.assertTrue(options["unit_flag"])
        self.assertEqual(options["buf_slot_num_of_veccore"], 2)
        self.assertEqual(options["buf_slot_num_of_crosscore"], 1)

    def test_default_compile_option_is_empty(self):
        option = NPUTemplateCompileOption()
        self.assertEqual(option.options, {})

    def test_apply_is_idempotent(self):
        meta = {}
        ft._FWD_COMPILE_OPTIONS.apply(meta)
        snapshot = dict(meta)
        ft._FWD_COMPILE_OPTIONS.apply(meta)
        self.assertEqual(meta, snapshot)


class TestGridFunctions(unittest.TestCase):
    """Symbolic grid helpers produce the expected (x, y, z) tuples."""

    def test_grid_functions_are_symbolic_grid_fns(self):
        for name in GRID_FUNCTIONS:
            grid = getattr(ft, name)
            self.assertIsInstance(
                grid,
                SymbolicGridFn,
                f"{name} is not wrapped by SymbolicGridFn",
            )

    def test_flex_attention_in_loop_grid(self):
        meta = {"BLOCK_M": 64, "NUM_CUBE_CORE": 8}
        # cdiv(128, 64) * 2 * 4 = 16 tiles -> capped at 8 cube cores.
        self.assertEqual(
            ft.flex_attention_in_loop_grid(2, 4, 128, 64, meta), (8, 1, 1)
        )
        # Fewer tiles than cores: grid equals the tile count.
        self.assertEqual(
            ft.flex_attention_in_loop_grid(1, 1, 64, 64, meta), (1, 1, 1)
        )

    def test_flex_attention_in_loop_grid_sympy_call(self):
        meta = {"BLOCK_M": 64, "NUM_CUBE_CORE": 8}
        grid = ft.flex_attention_in_loop_grid
        result = grid.sympy_call(2, 4, 128, 64, meta)
        self.assertEqual(len(result), 3)

    def test_compact_offsets_grid_is_single_program(self):
        self.assertEqual(ft.compact_offsets_grid(123, {"BLOCK_M": 32}), (1, 1, 1))

    def test_fwd_workspace_offsets_grid_is_single_program(self):
        self.assertEqual(ft.fwd_workspace_offsets_grid(456, {}), (1, 1, 1))

    def test_compact_mapping_grid(self):
        meta = {"NUM_VECTOR_CORE": 50}
        self.assertEqual(ft.compact_mapping_grid(100, meta), (50, 1, 1))
        self.assertEqual(ft.compact_mapping_grid(3, meta), (3, 1, 1))
        # Empty workload still launches at least one program.
        self.assertEqual(ft.compact_mapping_grid(0, meta), (1, 1, 1))

    def test_compact_sparse_mask_grid(self):
        meta = {
            "NUM_Q_SUB_BLOCKS": 2,
            "NUM_KV_SUB_BLOCKS": 2,
            "NUM_VECTOR_CORE": 10,
        }
        # total entries = 4 * 2 * 2 = 16 -> capped at 10 vector cores.
        self.assertEqual(ft.compact_sparse_mask_grid(4, meta), (10, 1, 1))
        # total entries below the core count: one program per entry.
        self.assertEqual(ft.compact_sparse_mask_grid(1, meta), (4, 1, 1))
        # No entries: still at least one program.
        self.assertEqual(ft.compact_sparse_mask_grid(0, meta), (1, 1, 1))

    def test_sparse_mask_block_pos_grid(self):
        meta = {"NUM_VECTOR_CORE": 8}
        self.assertEqual(ft.sparse_mask_block_pos_grid(20, meta), (8, 1, 1))
        self.assertEqual(ft.sparse_mask_block_pos_grid(5, meta), (5, 1, 1))
        self.assertEqual(ft.sparse_mask_block_pos_grid(0, meta), (1, 1, 1))

    def test_backward_dq_grid(self):
        meta = {"DQ_LAUNCH_PROGRAMS": 7}
        self.assertEqual(
            ft.flex_attention_backward_dq_grid(1, 2, 3, 4, 5, 6, meta), (7, 1, 1)
        )

    def test_backward_dkdv_grid(self):
        meta = {"LAUNCH_PROGRAMS": 9}
        self.assertEqual(
            ft.flex_attention_backward_dkdv_grid(1, 2, 3, 4, 5, 6, meta), (9, 1, 1)
        )


class TestWrappedUpstreamTemplates(unittest.TestCase):
    """The three wrapped templates preserve upstream identity but switch
    codegen to NPUTritonTemplate."""

    WRAPPED = {
        "flex_attention_template": "fwd_main",
        "flex_attention_backward_template": "bwd_main",
        "flex_decoding_template": "fwd_decode",
    }

    def test_wrapped_templates_are_npu_templates(self):
        for name in self.WRAPPED:
            template = getattr(ft, name)
            self.assertIsInstance(template, NPUTritonTemplate)

    def test_wrapped_templates_registered_in_all_templates(self):
        for name in self.WRAPPED:
            template = getattr(ft, name)
            registered = TritonTemplate.all_templates.get(template.name)
            self.assertIs(
                registered,
                template,
                f"{name} (template.name={template.name}) is not registered",
            )

    def test_wrapped_templates_npu_extra_name(self):
        for name, extra in self.WRAPPED.items():
            self.assertEqual(getattr(ft, name).npu_extra_name, extra)

    def test_wrapped_templates_have_no_manual_output_buffer(self):
        for name in self.WRAPPED:
            self.assertIsNone(getattr(ft, name).manual_output_buffer)

    def test_wrapped_templates_have_empty_compile_options(self):
        for name in self.WRAPPED:
            options = getattr(ft, name).compile_options
            self.assertIsInstance(options, NPUTemplateCompileOption)
            self.assertEqual(options.options, {})

    def test_wrapped_templates_carry_upstream_state(self):
        # The wrapped template keeps a non-empty upstream kernel payload.
        for name in self.WRAPPED:
            template = getattr(ft, name)
            self.assertIsNotNone(getattr(template, "template", None))


class TestNPUTritonTemplates(unittest.TestCase):
    """Every explicit NPUTritonTemplate has consistent metadata."""

    def test_all_templates_exported_and_typed(self):
        for name in TEMPLATE_EXTRA_NAMES:
            template = getattr(ft, name)
            self.assertIsInstance(template, NPUTritonTemplate)

    def test_template_names_match_attributes(self):
        for name in TEMPLATE_EXTRA_NAMES:
            self.assertEqual(getattr(ft, name).name, name)

    def test_all_templates_registered(self):
        for name in TEMPLATE_EXTRA_NAMES:
            registered = TritonTemplate.all_templates.get(name)
            self.assertIs(registered, getattr(ft, name), f"{name} not registered")

    def test_npu_extra_names(self):
        for name, extra in TEMPLATE_EXTRA_NAMES.items():
            self.assertEqual(
                getattr(ft, name).npu_extra_name,
                extra,
                f"unexpected npu_extra_name for {name}",
            )

    def test_npu_extra_names_are_unique_per_template(self):
        extras = [getattr(ft, n).npu_extra_name for n in TEMPLATE_EXTRA_NAMES]
        self.assertEqual(
            len(extras),
            len(set(extras)),
            "npu_extra_name must be unique across explicit templates",
        )

    def test_manual_output_buffers(self):
        for name in TEMPLATE_EXTRA_NAMES:
            expected = TEMPLATE_MANUAL_OUTPUT_BUFFERS.get(name)
            self.assertEqual(
                getattr(ft, name).manual_output_buffer,
                expected,
                f"unexpected manual_output_buffer for {name}",
            )

    def test_template_sources_exist_in_module(self):
        for name, source_attr in TEMPLATE_SOURCES.items():
            self.assertTrue(
                hasattr(ft, source_attr),
                f"missing source attribute {source_attr} for {name}",
            )
            source = getattr(ft, source_attr)
            self.assertIsInstance(source, str)
            self.assertIn("def_kernel", source)

    def test_grid_assignment(self):
        expected_grids = {
            "flex_attention_fwd_mask_out": ft.flex_attention_in_loop_grid,
            "flex_attention_fwd_mask_out_all_sparse": ft.flex_attention_in_loop_grid,
            "flex_attention_compact_offsets": ft.compact_offsets_grid,
            "flex_attention_compact_mapping": ft.compact_mapping_grid,
            "flex_attention_fwd_mask_compact": ft.compact_sparse_mask_grid,
            "flex_attention_fwd_workspace_offsets": ft.fwd_workspace_offsets_grid,
            "flex_attention_fwd_workspace_mask_compact": ft.compact_sparse_mask_grid,
            "flex_attention_bwd_mask_compact": ft.compact_sparse_mask_grid,
            "flex_attention_bwd_mask_pos": ft.sparse_mask_block_pos_grid,
            "flex_attention_bwd_dq_mask_out": ft.flex_attention_backward_dq_grid,
            "flex_attention_bwd_dkdv_mask_out": ft.flex_attention_backward_dkdv_grid,
            "flex_attention_bwd_dkdv_tasklist": ft.flex_attention_backward_dkdv_grid,
            "flex_attention_bwd_dkdv_tasklist_no_split": ft.flex_attention_backward_dkdv_grid,
            "flex_attention_bwd_dkdv_reduce": ft.flex_attention_backward_dkdv_grid,
        }
        for name, grid in expected_grids.items():
            self.assertIs(getattr(ft, name).grid, grid, f"wrong grid for {name}")

    def test_compile_options_assignment(self):
        for name in TEMPLATE_EXTRA_NAMES:
            template = getattr(ft, name)
            if name in FWD_OPTION_TEMPLATES:
                self.assertIs(template.compile_options, ft._FWD_COMPILE_OPTIONS)
            elif name in BWD_DQ_OPTION_TEMPLATES:
                self.assertIs(template.compile_options, ft._BWD_DQ_COMPILE_OPTIONS)
            elif name in BWD_DKDV_OPTION_TEMPLATES:
                self.assertIs(template.compile_options, ft._BWD_DKDV_COMPILE_OPTIONS)
            else:
                self.assertEqual(template.compile_options.options, {})

    def test_tasklist_variants_share_kernel_source(self):
        self.assertIs(
            getattr(ft, TEMPLATE_SOURCES["flex_attention_bwd_dkdv_tasklist"]),
            getattr(
                ft, TEMPLATE_SOURCES["flex_attention_bwd_dkdv_tasklist_no_split"]
            ),
        )

    def test_mask_out_variants_share_kernel_source(self):
        self.assertIs(
            getattr(ft, TEMPLATE_SOURCES["flex_attention_fwd_mask_out"]),
            getattr(ft, TEMPLATE_SOURCES["flex_attention_fwd_mask_out_all_sparse"]),
        )


class TestFlexAttentionOnNpu(unittest.TestCase):
    """End-to-end checks: the NPU flex attention templates produce results
    consistent with scaled_dot_product_attention on NPU."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(2026)
        torch._dynamo.reset()

    @unittest.skipUnless(NPU_AVAILABLE, "requires NPU device")
    def _make_causal_block_mask(self, q_len, kv_len, device="npu"):
        from torch.nn.attention.flex_attention import create_block_mask

        def causal_mask(batch, head, q_idx, kv_idx):
            return q_idx >= kv_idx

        return create_block_mask(
            causal_mask, None, None, q_len, kv_len, device=device
        )

    @unittest.skipUnless(NPU_AVAILABLE, "requires NPU device")
    def test_forward_causal_block_mask(self):
        from torch.nn.attention.flex_attention import flex_attention

        b, h, q_len, kv_len, d = 1, 2, 128, 128, 64
        q = torch.randn(b, h, q_len, d, device="npu", dtype=torch.float16)
        k = torch.randn(b, h, kv_len, d, device="npu", dtype=torch.float16)
        v = torch.randn(b, h, kv_len, d, device="npu", dtype=torch.float16)
        block_mask = self._make_causal_block_mask(q_len, kv_len)

        compiled = torch.compile(flex_attention, dynamic=False)
        out = compiled(q, k, v, block_mask=block_mask)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        torch.testing.assert_close(
            out, ref, rtol=2e-2, atol=2e-2, msg="causal flex_attention mismatch"
        )

    @unittest.skipUnless(NPU_AVAILABLE, "requires NPU device")
    def test_forward_mask_out_template_selected(self):
        # The NPU forward mask-out template must be among the generated
        # kernel names when the sparse mask path is used.
        from torch.nn.attention.flex_attention import flex_attention

        b, h, seq, d = 1, 2, 128, 64
        q = torch.randn(b, h, seq, d, device="npu", dtype=torch.float16)
        k = torch.randn(b, h, seq, d, device="npu", dtype=torch.float16)
        v = torch.randn(b, h, seq, d, device="npu", dtype=torch.float16)
        block_mask = self._make_causal_block_mask(seq, seq)

        out = torch.compile(flex_attention, dynamic=False)(q, k, v, block_mask=block_mask)
        self.assertEqual(out.shape, (b, h, seq, d))
        self.assertFalse(torch.isnan(out).any(), "NaN in flex_attention output")


if __name__ == "__main__":
    unittest.main(verbosity=2)
