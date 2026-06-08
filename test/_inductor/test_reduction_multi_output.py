"""
Regression tests for NPU Inductor Triton Kernel DSL generation bugs.

Covers the following fixed issues:
  Bug 1: NameError('tmp15 is not defined') — reduction result variable used in
         non-reduction store placed inside the loop instead of outside.
  Bug 2: NameError('ps1 is not defined') — CSE variable definition order error
         in dynamic shape mode.
  Bug 3: NameError('x1'/'x3' is not defined') — upstream outside_loop_vars
         pollution causing misplacement of stores.
  Bug 4: IndentationError / NameError regression — IndentedBuffer _indent
         inconsistency and reduction_result_vars cleared across nodes.

Key trigger condition: A fused kernel containing a persistent reduction whose
result is referenced by a subsequent (epilogue) node's store operation.
This exercises the reduction_result_vars + _deferred_reduction_stores path.
"""

import torch
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu


class TestReductionMultiOutput(TestUtils):
    """
    Tests for reduction kernels where the reduction result variable is stored
    to multiple output buffers (one via store_reduction, one or more via regular
    store referencing the same reduction result).

    This directly exercises the fix for:
      - reduction_result_vars independent set (avoids outside_loop_vars pollution)
      - _deferred_reduction_stores plain list (avoids IndentedBuffer _indent issues)
      - deferred output only in last_tiling branch (correct position in static/dynamic)
      - reduction_result_vars NOT cleared at end of codegen_body (cross-node survival)
    """

    # ------------------------------------------------------------------ #
    #  Test Group 1: Reduction result stored to two outputs (static mode) #
    # ------------------------------------------------------------------ #

    def _reduction_dual_store(self, x, dim):
        """Sum reduction, result stored to two separate outputs."""
        r = x.sum(dim)
        # The second return value references the same reduction result 'r',
        # triggering the deferred store path in codegen.
        return r, r

    @parametrize("shape", [(128, 38), (64, 128, 16), (32, 8, 128, 4)])
    @parametrize("dim", [(-1,), (1,)])
    @parametrize("dtype", ["float32"])
    def test_reduction_dual_store_static(self, shape, dim, dtype):
        """
        Bug #1 / #3 / #4 regression: reduction result 'tmp15' used in a second
        store (out_ptr1) must appear AFTER the reduction definition and with
        correct indentation (outside the reduction loop).
        Uses small reduction dimension (e.g., 38) to force loops_r2 > 1 when
        R2BLOCK_SUB=4, which was the exact trigger for op49_op321.
        """
        input_tensor = self._generate_tensor(shape, dtype)

        ref_out0, ref_out1 = self._reduction_dual_store(input_tensor, *dim)

        compiled_fn = torch.compile(
            self._reduction_dual_store, backend="inductor", dynamic=False
        )
        ind_out0, ind_out1 = compiled_fn(input_tensor, *dim)

        self.assertEqual(ref_out0, ind_out0, atol=1e-1, rtol=1e-1)
        self.assertEqual(ref_out1, ind_out1, atol=1e-1, rtol=1e-1)

    # ------------------------------------------------------------------ #
    #  Test Group 2: Same pattern with dynamic=True                     #
    # ------------------------------------------------------------------ #

    @parametrize("shape", [(128, 38), (64, 128, 16), (32, 8, 128, 4)])
    @parametrize("dim", [(-1,), (1,)])
    @parametrize("dtype", ["float32"])
    def test_reduction_dual_store_dynamic(self, shape, dim, dtype):
        """
        Bug #2 / #4 regression: dynamic shape mode must also correctly place
        the deferred store outside the loop with proper indentation.
        Also exercises the ps* variable ordering fix in wrapper.py.
        """
        input_tensor = self._generate_tensor(shape, dtype)

        ref_out0, ref_out1 = self._reduction_dual_store(input_tensor, *dim)

        compiled_fn = torch.compile(
            self._reduction_dual_store, backend="inductor", dynamic=True
        )
        ind_out0, ind_out1 = compiled_fn(input_tensor, *dim)

        self.assertEqual(ref_out0, ind_out0, atol=1e-1, rtol=1e-1)
        self.assertEqual(ref_out1, ind_out1, atol=1e-1, rtol=1e-1)

    # ------------------------------------------------------------------ #
    #  Test Group 3: Reduction + epilogue (unsqueeze/broadcast)          #
    # ------------------------------------------------------------------ #

    def _reduction_with_epilogue(self, x, dim):
        """Reduction followed by an epilogue op that uses the result."""
        r = x.sum(dim)
        # unsqueeze creates a new node whose store references 'r'
        v = r.unsqueeze(dim[0] if isinstance(dim, tuple) else dim)
        return r, v

    @parametrize("shape", [(128, 38), (256, 64, 10), (16, 16, 64, 8)])
    @parametrize("dim", [(-1,), (1,)])
    @parametrize("dtype", ["float32"])
    def test_reduction_epilogue_static(self, shape, dim, dtype):
        """
        Regression for the full op49_op321 pattern:
        - reduction node generates tmp15 via tl.sum()
        - epilogue (pointwise) node stores tmp15 to out_ptr1
        - Both stores must be outside the reduction loop with matching indent.
        """
        input_tensor = self._generate_tensor(shape, dtype)

        ref_r, ref_v = self._reduction_with_epilogue(input_tensor, *dim)

        compiled_fn = torch.compile(
            self._reduction_with_epilogue, backend="inductor", dynamic=False
        )
        ind_r, ind_v = compiled_fn(input_tensor, *dim)

        self.assertEqual(ref_r, ind_r, atol=1e-1, rtol=1e-1)
        self.assertEqual(ref_v, ind_v, atol=1e-1, rtol=1e-1)

    @parametrize("shape", [(128, 38), (256, 64, 10), (16, 16, 64, 8)])
    @parametrize("dim", [(-1,), (1,)])
    @parametrize("dtype", ["float32"])
    def test_reduction_epilogue_dynamic(self, shape, dim, dtype):
        """Same as above but in dynamic mode."""
        input_tensor = self._generate_tensor(shape, dtype)

        ref_r, ref_v = self._reduction_with_epilogue(input_tensor, *dim)

        compiled_fn = torch.compile(
            self._reduction_with_epilogue, backend="inductor", dynamic=True
        )
        ind_r, ind_v = compiled_fn(input_tensor, *dim)

        self.assertEqual(ref_r, ind_r, atol=1e-1, rtol=1e-1)
        self.assertEqual(ref_v, ind_v, atol=1e-1, rtol=1e-1)

    # ------------------------------------------------------------------ #
    #  Test Group 4: Various reduction types                             #
    # ------------------------------------------------------------------ #

    def _reduction_mean_dual(self, x, dim):
        """Mean reduction with dual output."""
        m = x.mean(dim)
        return m, m

    @parametrize("shape", [(128, 50), (64, 100, 12)])
    @parametrize("dim", [(-1,), (1,)])
    @parametrize("dtype", ["float32"])
    def test_mean_dual_store_static(self, shape, dim, dtype):
        """Mean reduction (non-sum) with dual store, static mode."""
        input_tensor = self._generate_tensor(shape, dtype)

        ref0, ref1 = self._reduction_mean_dual(input_tensor, *dim)

        compiled_fn = torch.compile(
            self._reduction_mean_dual, backend="inductor", dynamic=False
        )
        ind0, ind1 = compiled_fn(input_tensor, *dim)

        self.assertEqual(ref0, ind0, atol=1e-1, rtol=1e-1)
        self.assertEqual(ref1, ind1, atol=1e-1, rtol=1e-1)

    def _reduction_var_mean_dual(self, x, dim):
        """Var_mean reduction with dual output on both results."""
        var, mean = x.var_mean(dim)
        return var, mean, var, mean

    @parametrize("shape", [(64, 48), (32, 32, 20)])
    @parametrize("dim", [(-1,)])
    @parametrize("dtype", ["float32"])
    def test_var_mean_multi_store_static(self, shape, dim, dtype):
        """
        Var_mean produces TWO reduction results (var and mean), each stored
        twice. Exercises multiple entries in reduction_result_vars.
        """
        input_tensor = self._generate_tensor(shape, dtype)

        ref = self._reduction_var_mean_dual(input_tensor, *dim)

        compiled_fn = torch.compile(
            self._reduction_var_mean_dual, backend="inductor", dynamic=False
        )
        ind = compiled_fn(input_tensor, *dim)

        for i in range(len(ref)):
            self.assertEqual(ref[i], ind[i], atol=1e-1, rtol=1e-1)

    # ------------------------------------------------------------------ #
    #  Test Group 5: Edge cases — shapes that exercise different          #
    #               codegen_range branches                               #
    # ------------------------------------------------------------------ #

    def _sum_3d_last_dim(self, x):
        """3D sum over last dim, result used in two stores."""
        s = x.sum(2)
        return s, s

    @parametrize("shape", [(8, 128, 38), (4, 64, 100), (2, 32, 200)])
    @parametrize("dtype", ["float32"])
    def test_3d_reduction_last_dim_static(self, shape, dtype):
        """
        3D tensor with reduction on last dimension. Exercises tiling_axis +
        reduction axis combination where the reduction axis is NOT the last
        tiling axis, potentially hitting a different codegen_range branch.
        """
        input_tensor = self._generate_tensor(shape, dtype)

        ref0, ref1 = self._sum_3d_last_dim(input_tensor)

        compiled_fn = torch.compile(
            self._sum_3d_last_dim, backend="inductor", dynamic=False
        )
        ind0, ind1 = compiled_fn(input_tensor)

        self.assertEqual(ref0, ind0, atol=1e-1, rtol=1e-1)
        self.assertEqual(ref1, ind1, atol=1e-1, rtol=1e-1)

    @parametrize("shape", [(8, 128, 38), (4, 64, 100), (2, 32, 200)])
    @parametrize("dtype", ["float32"])
    def test_3d_reduction_last_dim_dynamic(self, shape, dtype):
        """Same 3D pattern but in dynamic mode."""
        input_tensor = self._generate_tensor(shape, dtype)

        ref0, ref1 = self._sum_3d_last_dim(input_tensor)

        compiled_fn = torch.compile(
            self._sum_3d_last_dim, backend="inductor", dynamic=True
        )
        ind0, ind1 = compiled_fn(input_tensor)

        self.assertEqual(ref0, ind0, atol=1e-1, rtol=1e-1)
        self.assertEqual(ref1, ind1, atol=1e-1, rtol=1e-1)

    def _sum_4d_mid_dim(self, x):
        """4D sum over a middle dimension, dual store."""
        s = x.sum(1)
        return s, s

    @parametrize("shape", [(16, 38, 64, 8), (8, 50, 32, 4)])
    @parametrize("dtype", ["float32"])
    def test_4d_reduction_mid_dim_static(self, shape, dtype):
        """
        4D tensor reducing a middle dimension. This exercises the axis
        reordering logic in codegen_body() where non-tiling axes are moved
        to the front of sorted_axis.
        """
        input_tensor = self._generate_tensor(shape, dtype)

        ref0, ref1 = self._sum_4d_mid_dim(input_tensor)

        compiled_fn = torch.compile(
            self._sum_4d_mid_dim, backend="inductor", dynamic=False
        )
        ind0, ind1 = compiled_fn(input_tensor)

        self.assertEqual(ref0, ind0, atol=1e-1, rtol=1e-1)
        self.assertEqual(ref1, ind1, atol=1e-1, rtol=1e-1)


instantiate_parametrized_tests(TestReductionMultiOutput)

if __name__ == "__main__":
    run_tests()
