# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Unit tests for the provable-clamp proof engine (pure functions).

The clamp skip's correctness lives entirely in two pure functions:

- ``_npu_index_provably_nonnegative(expr, lb, axis_names)`` — linear
  lower-bound proof that ``expr >= 0`` on kept lanes given axis lower bounds;
- ``_npu_mask_lower_atoms(mask_name, defs, axis_names)`` — resolving an
  emitted tmp-mask DSL definition back into ``{axis: lower_bound}`` atoms.

Axis identity is STRUCTURAL and single-sourced: ``axis_names`` is the
kernel range tree's symbol set; a name not in the set is not an axis, and
an empty set makes every proof decline (clamp kept) — no naming-convention
fallback exists. yolov3's head sites are the canonical proven form
``-4 + x0`` under a mask implying ``x0 >= 4``.

Execution model: the real suite lives under ``__name__ == "__main__"`` and
runs as a subprocess; the pytest surface is ONE wrapper test. Empirically,
any prior test module that EXECUTED tests in a shared pytest process
poisons later multi-compile files (compile worker #2+ dies with "Cannot
re-initialize NPU in forked subprocess"; imports alone are harmless). The
subprocess makes this file order-immune and keeps the heavy backend
imports out of the pytest process entirely. No environment variable is
used — the outer/inner split is just ``__name__``.
"""

import os
import subprocess
import sys
import unittest

if __name__ == "__main__":
    # Subprocess entry (python <this file>): define and run the real suite.
    import sympy
    import torch  # noqa: F401
    import torch_npu  # noqa: F401

    from torch_npu._inductor.triton_experimental.codegen.triton import (
        _npu_index_provably_nonnegative,
        _npu_mask_lower_atoms,
        _npu_body_defs,
    )

    x0, x1, r0, tmp0, i0 = sympy.symbols("x0 x1 r0 tmp0 i0")

    class _MockCompute:
        """get_lines_ref returns str()-able lines, as the real code does."""

        def __init__(self, lines):
            self._lines = list(lines)

        def get_lines_ref(self):
            return self._lines

    class _MockKernel:
        def __init__(self, lines):
            self.compute = _MockCompute(lines)

    # Structural axis set used by the pure-function cases.
    AXES = {"x0", "x1", "r0"}

    class ProvablyNonnegativeCases(unittest.TestCase):
        def test_yolov3_canonical_proven(self):
            self.assertTrue(
                _npu_index_provably_nonnegative(-4 + x0, {"x0": 4}, AXES)
            )

        def test_insufficient_bound_keeps_clamp(self):
            self.assertFalse(
                _npu_index_provably_nonnegative(-4 + x0, {"x0": 2}, AXES)
            )

        def test_nonnegative_origin_always_proven(self):
            self.assertTrue(
                _npu_index_provably_nonnegative(2 + x0 + r0, {}, AXES)
            )

        def test_negative_coefficient_rejected(self):
            self.assertFalse(
                _npu_index_provably_nonnegative(x0 - 2 * x1, {"x0": 0}, AXES)
            )

        def test_tmp_indirect_symbol_rejected(self):
            # TMP rejection is independent of set membership.
            self.assertFalse(
                _npu_index_provably_nonnegative(
                    tmp0 + x0, {"x0": 4}, AXES | {"tmp0"})
            )

    class MaskLowerAtomsCases(unittest.TestCase):
        def test_nested_named_conjunction(self):
            k = _MockKernel([
                "tmp0 = x0 >= 4",
                "tmp1 = r0 >= 0",
                "mask = tmp0 & tmp1",
            ])
            atoms = _npu_mask_lower_atoms(
                "mask", _npu_body_defs(k), {"x0", "r0"})
            self.assertEqual(atoms.get("x0"), 4)
            self.assertEqual(atoms.get("r0"), 0)

        def test_grammar_wrappers_unwrapped(self):
            k = _MockKernel([
                "t0 = tl.full([1], 4, tl.int32)",
                "tmp0 = tl.broadcast_to(x0.to(tl.int32), [8]) >= t0",
                "mask = tmp0 & (x1 >= 0)",
            ])
            atoms = _npu_mask_lower_atoms(
                "mask", _npu_body_defs(k), {"x0", "x1"})
            self.assertEqual(atoms.get("x0"), 4)
            self.assertEqual(atoms.get("x1"), 0)

        def test_strongest_bound_wins(self):
            k = _MockKernel([
                "tmp0 = x0 >= 2",
                "tmp1 = x0 >= 4",
                "mask = tmp0 & tmp1",
            ])
            atoms = _npu_mask_lower_atoms("mask", _npu_body_defs(k), {"x0"})
            self.assertEqual(atoms.get("x0"), 4)

        def test_upper_bound_only_contributes_nothing(self):
            k = _MockKernel(["mask = x0 < 512"])
            atoms = _npu_mask_lower_atoms("mask", _npu_body_defs(k), {"x0"})
            self.assertEqual(atoms, {})

        def test_empty_axis_set_declines_everything(self):
            # No structural source -> no axis -> no atoms (sound decline).
            k = _MockKernel(["mask = x0 >= 4"])
            atoms = _npu_mask_lower_atoms("mask", _npu_body_defs(k), set())
            self.assertEqual(atoms, {})

    class StructuralAxisIdentityCases(unittest.TestCase):
        def test_structural_accepts_upstream_index_naming(self):
            # "i0" is upstream's INDEX prefix — accepted because the SET says
            # so; upstream renaming cannot blind the prover.
            self.assertTrue(
                _npu_index_provably_nonnegative(
                    -4 + i0, {"i0": 4}, {"i0", "x0"})
            )

        def test_structural_rejects_symbol_outside_set(self):
            self.assertFalse(
                _npu_index_provably_nonnegative(i0 + x0, {"x0": 4}, {"x0"})
            )

        def test_dsl_derived_name_via_structural_prefix(self):
            k = _MockKernel([
                "i0index = i0offset + tl.arange(0, BLOCK)",
                "tmp0 = i0index >= 4",
                "mask = tmp0",
            ])
            atoms = _npu_mask_lower_atoms("mask", _npu_body_defs(k), {"i0"})
            self.assertEqual(atoms.get("i0"), 4)

    unittest.main()

else:
    # Pytest surface: one wrapper test running the suite as a subprocess.
    class TestProvableClampProverSuite(unittest.TestCase):
        def test_prover_suite(self):
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__)],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            self.assertEqual(
                proc.returncode, 0, "inner suite failed (see output above)"
            )
