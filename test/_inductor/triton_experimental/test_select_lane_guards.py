# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Regression coverage for the three select-lane rewrite guards.

``NPU_SELECT_EXTRACT_SLICE`` rewrites a stride-k select/slice load into a
contiguous full load plus a register-level ``extract_slice``. Widening the
inner slot is only sound while that slot carries no other real extent, so the
rewrite bails out on three structural facts captured at load() time:

  * ``addr_real_inner``  - the load address already has a real inner extent
    (IndexingOptions ``expand_shape``, or a CSE temporary's block shape),
  * foreign base axis    - the address minus the ``k*inner`` term references an
    axis that owns no tensor slot of this kernel's layouts,
  * foreign mask         - a mask term other than the replaced inner mask has a
    real inner extent.

The first block pins each predicate on its own. The second drives
``_maybe_rewrite_select_lane_load`` on a synthetic kernel to show that every
guard is wired to the rewrite (the load is left alone) and that the unguarded
path still emits the contiguous ``extract_slice`` load. The last test compiles
a real graph, so the safe path keeps its end-to-end coverage.
"""

import os
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import sympy
import torch
from torch._inductor.codegen.triton import IndexingOptions, TritonKernel
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._ordered_set import OrderedSet

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.triton_experimental.codegen import triton as npu_triton_codegen

NPUTritonKernel = npu_triton_codegen.NPUTritonKernel

x0, x1, y0 = sympy.symbols("x0 x1 y0")


def _bare_kernel(**attrs):
    """An NPUTritonKernel carrying only the attributes the guards read."""
    kernel = object.__new__(NPUTritonKernel)
    kernel.cse = SimpleNamespace(varname_map={})
    kernel.range_trees = []
    for name, value in attrs.items():
        setattr(kernel, name, value)
    return kernel


class TestSelectLaneGuards(TestCase):
    # -- guard 1: CSE block shape / address inner extent ------------------
    def test_shape_inner_extent_is_one(self):
        cases = (
            (None, False),  # unknown shape must stay unsafe
            ((), True),  # 0-d tile
            ((1,), True),
            (("XBLOCK", 1), True),
            ((sympy.Integer(1),), True),
            (("XBLOCK",), False),
            (("XBLOCK", "4"), False),
        )
        for shape, expected in cases:
            with self.subTest(shape=shape):
                self.assertEqual(
                    NPUTritonKernel._shape_inner_extent_is_one(shape), expected
                )

    def test_addr_real_inner_reads_expand_shape(self):
        cases = (
            (("XBLOCK",), True),  # 1-d dense address: the inner slot IS XBLOCK
            (("XBLOCK", 1), False),
            (("XBLOCK", "4"), True),
        )
        kernel = _bare_kernel()
        for expand_shape, expected in cases:
            with self.subTest(expand_shape=expand_shape):
                options = SimpleNamespace(
                    expand_str="[XBLOCK, 1]", expand_shape=expand_shape
                )
                self.assertEqual(
                    kernel._select_lane_addr_has_real_inner(options, 2 * x0),
                    expected,
                )

    def test_addr_real_inner_reads_cse_shapes(self):
        tmp0 = sympy.Symbol("tmp0")
        options = SimpleNamespace(expand_str=None, expand_shape=None)
        cases = (
            (SimpleNamespace(shape=["XBLOCK", "4"]), True),
            (SimpleNamespace(shape=["XBLOCK", "1"]), False),
            (SimpleNamespace(shape=None), True),  # unknown shape -> unsafe
        )
        for var, expected in cases:
            with self.subTest(shape=getattr(var, "shape", None)):
                kernel = _bare_kernel(cse=SimpleNamespace(varname_map={"tmp0": var}))
                self.assertEqual(
                    kernel._select_lane_addr_has_real_inner(options, 2 * tmp0),
                    expected,
                )
        # A symbol that is not a CSE variable carries no block shape.
        kernel = _bare_kernel()
        self.assertFalse(kernel._select_lane_addr_has_real_inner(options, 2 * tmp0))

    # -- guard 2: foreign axis in the widened base address -----------------
    def test_base_index_has_foreign_axis(self):
        slots = {"x0", "x1"}
        cases = (
            (sympy.Integer(0), False),
            (128 * x0, False),
            (128 * x0 + 2 * x1, False),  # inner term still allowed
            (128 * x0 + 16 * y0, True),  # y0 owns no slot here
            (y0, True),
        )
        for base_index, expected in cases:
            with self.subTest(base_index=base_index):
                self.assertEqual(
                    NPUTritonKernel._select_lane_base_index_has_foreign_axis(
                        base_index, "x1", slots
                    ),
                    expected,
                )

    # -- guard 3: other mask terms with a real inner extent ----------------
    def test_mask_has_real_inner(self):
        kernel = _bare_kernel(
            range_trees=[SimpleNamespace(mask_name=lambda: "x0mask")],
            cse=SimpleNamespace(
                varname_map={"tmp1": SimpleNamespace(shape=["XBLOCK", "4"])}
            ),
        )
        has_real_inner = kernel._select_lane_mask_has_real_inner
        self.assertFalse(has_real_inner((), "x1", "x1mask"))
        self.assertFalse(has_real_inner(("x0mask",), "x1", "x1mask"))
        self.assertFalse(has_real_inner(("x1mask",), "x1", "x1mask"))
        self.assertFalse(
            has_real_inner(
                (SimpleNamespace(shape=["XBLOCK", "1"]),), "x1", "x1mask"
            )
        )
        self.assertTrue(has_real_inner(("tmp1",), "x1", "x1mask"))
        self.assertTrue(has_real_inner(("x1",), "x1", "x1mask"))

    # -- the recorded facts are what the rewrite reads ---------------------
    def test_recording_captures_guard_facts(self):
        kernel = object.__new__(NPUTritonKernel)
        kernel._npu_linearize = True
        kernel.inside_reduction = False
        kernel._npu_select_lane_loads = {}
        kernel.cse = SimpleNamespace(varname_map={})
        kernel.range_trees = []
        kernel.range_tree_nodes = {
            x0: SimpleNamespace(root=SimpleNamespace(is_reduction=False))
        }
        kernel.args = SimpleNamespace(input=lambda name: f"in_ptr_{name}")
        result_var = SimpleNamespace(name="tmp0", mask_vars=("x0mask",))
        with mock.patch.object(npu_triton_codegen.ncfg, "select_extract_slice", True):
            kernel._maybe_record_select_lane_load(
                "in0",
                result_var,
                2 * x0,
                prepared_index=2 * x0,
                options=SimpleNamespace(
                    expand_str="[XBLOCK, 1]", expand_shape=("XBLOCK", "4")
                ),
            )
        target = kernel._npu_select_lane_loads["tmp0"]
        self.assertEqual(target["lane"], 2)
        self.assertEqual(target["pointer"], "in_ptr_in0")
        self.assertTrue(target["addr_real_inner"])
        self.assertEqual(target["mask_terms"], ["x0mask"])

    def test_indexing_capture_tracks_the_emitted_options(self):
        # The select-lane guards read the captured IndexingOptions. The
        # linearize tail below can wrap the address in a dense
        # ``tl.broadcast_to`` after ``load()`` captured
        # ``super().indexing()``'s result, so the capture must follow the
        # object that is actually emitted; otherwise the "address was
        # broadcast to a real inner block" guard reads a stale expand_str=None.
        index = 2 * x0
        upstream_result = IndexingOptions(
            "2*x0", OrderedSet(), None, False, index, expand_shape=("XBLOCK",)
        )
        kernel = object.__new__(NPUTritonKernel)
        kernel._npu_linearize = True
        kernel.range_trees = [SimpleNamespace(numel=4, is_reduction=False)]
        kernel.cse = SimpleNamespace(varname_map={})
        kernel._load_mask = "tmp0mask"
        kernel._npu_capture_prepared_load_index = True
        kernel._npu_prepared_load_options = None
        with (
            mock.patch.object(TritonKernel, "indexing", return_value=upstream_result),
            mock.patch.object(npu_triton_codegen, "triton_codegen_linearize", True),
            mock.patch.object(kernel, "filter_masks"),
            mock.patch.object(kernel, "dense_size_str", return_value="[XBLOCK]"),
            mock.patch.object(
                npu_triton_codegen,
                "_masked_index_needs_nonnegative_clamp",
                lambda *args, **kwargs: False,
            ),
            mock.patch.object(
                npu_triton_codegen.ncfg, "skip_redundant_broadcast", False
            ),
        ):
            emitted = kernel.indexing(index)

        self.assertIsNot(emitted, upstream_result)
        self.assertIn("tl.broadcast_to", emitted.index_str)
        self.assertIs(kernel._npu_prepared_load_options, emitted)
        # The stale object reports "no broadcast" (expand_str is None), the
        # emitted one carries the real dense inner block. Only the latter must
        # veto the rewrite.
        self.assertFalse(
            kernel._select_lane_addr_has_real_inner(upstream_result, index)
        )
        self.assertTrue(kernel._select_lane_addr_has_real_inner(emitted, index))

    # -- wiring: each guard actually stops the rewrite ---------------------
    def _rewrite_kernel(self, lines, targets):
        kernel = object.__new__(NPUTritonKernel)
        kernel.body = SimpleNamespace(_lines=list(lines))
        kernel.range_trees = [
            SimpleNamespace(
                is_reduction=False,
                var_tensor_dims={"x0": 0, "x1": 1},
                node_block_constexpr={"x0": "XBLOCK", "x1": "YBLOCK"},
                tree_node_mapping={},
                mask_name=lambda: "x0mask",
            )
        ]
        kernel.cse = SimpleNamespace(varname_map={})
        kernel._npu_select_lane_loads = targets
        kernel._npu_linearize = True
        kernel._linearize_applied = True
        kernel.inside_reduction = False
        kernel._npu_select_slice_done = False
        kernel.overrides = SimpleNamespace(extract_slice=lambda *args: "EXTRACT_SLICE")
        kernel.triton_tensor_ndim = lambda: 2
        kernel.indexing_size_str = lambda slot: ""
        kernel.index_to_str = lambda index: str(index)
        return kernel

    def _run_rewrite(self, kernel):
        with (
            mock.patch.object(npu_triton_codegen.ncfg, "select_extract_slice", True),
            mock.patch.object(
                npu_triton_codegen.ncfg, "select_extract_slice_strided", True
            ),
        ):
            kernel._maybe_rewrite_select_lane_load()
        return "\n".join(
            line for line in kernel.body._lines if isinstance(line, str)
        )

    def _case_a_kernel(self, addr_real_inner):
        line = "tmp0 = tl.load(in_ptr0 + (8*x0), xmask, other=0.0)"
        target = {
            "lane": 4,
            "index": 8 * x0,
            "pointer": "in_ptr0",
            "addr_real_inner": addr_real_inner,
            "mask_terms": (),
        }
        return self._rewrite_kernel([line], {"tmp0": target}), line

    def _case_b_kernel(self, addr_real_inner=False):
        line = "tmp0 = tl.load(in_ptr0 + (2*x1 + 128*x0), x1mask, other=0.0)"
        target = {
            "lane": 2,
            "index": 2 * x1 + 128 * x0,
            "pointer": "in_ptr0",
            "addr_real_inner": addr_real_inner,
            "mask_terms": ("x0mask",),
        }
        return self._rewrite_kernel([line], {"tmp0": target}), line

    def test_select_lane_rewrite_fires_when_no_guard_trips(self):
        # Case A (select: inner axis broadcast away) widens the address.
        kernel, _ = self._case_a_kernel(addr_real_inner=False)
        code = self._run_rewrite(kernel)
        self.assertIn("_es_lane0 = tl.arange(0, 4)", code)
        self.assertIn("EXTRACT_SLICE", code)

        # Case B (x[:, ::k]) widens the inner arange to k*block and keeps a
        # tail-bounded mask.
        kernel, _ = self._case_b_kernel()
        code = self._run_rewrite(kernel)
        self.assertIn("_es_slane0 = tl.arange(0, 2*YBLOCK)", code)
        self.assertIn("2*x1numel", code)
        self.assertIn("_es_sfull0", code)
        self.assertIn("EXTRACT_SLICE", code)

    def test_addr_real_inner_guard_skips_rewrite(self):
        kernel, line = self._case_a_kernel(addr_real_inner=True)
        self.assertEqual(self._run_rewrite(kernel), line)

    def test_foreign_base_axis_guard_skips_rewrite(self):
        kernel, line = self._case_b_kernel()
        with mock.patch.object(
            NPUTritonKernel,
            "_select_lane_base_index_has_foreign_axis",
            return_value=True,
        ):
            self.assertEqual(self._run_rewrite(kernel), line)

    def test_foreign_mask_guard_skips_rewrite(self):
        kernel, line = self._case_b_kernel()
        with mock.patch.object(
            NPUTritonKernel,
            "_select_lane_mask_has_real_inner",
            return_value=True,
        ):
            self.assertEqual(self._run_rewrite(kernel), line)

    # -- end to end: the safe path keeps the optimization ------------------
    def test_safe_path_rewrites_a_real_graph(self):
        if not torch.npu.is_available():
            self.skipTest("requires an NPU")

        def fn(t):
            return t[:, ::2] + 1.0

        x = torch.randn(64, 256, device="npu")
        compiled = torch.compile(fn, options={"npu_backend": "triton_experimental"})
        actual, codes = run_and_get_code(compiled, x)
        source = "\n".join(codes)
        self.assertIn("extract_slice(", source)
        self.assertIn("_es_sfull", source)
        torch.testing.assert_close(actual, fn(x))


if __name__ == "__main__":
    run_tests()
