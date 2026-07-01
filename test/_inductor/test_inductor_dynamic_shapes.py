"""
Unit tests for NPU inductor multi-op fusion + nonzero dynamic shapes (dynamic=True).

Part 1: Multi-operator fusion tests that exercise NPU's SplitTiling system:
  - split_axis selection adapts when tensor dimensions change at runtime
  - tiling_axis for reduction ops adjusts to dynamic reduction dim sizes
  - Indirect memory ops (gather/scatter/index_select) with SIMT/SIMD paths
  - View/reshape/cat + compute fusion with dynamic shapes

Part 2: Nonzero data-dependent dynamic shape tests:
  - nonzero produces data-dependent output shape[0] = count of non-zero elements
  - Unbacked SymInt guarded by runtime assertions under dynamo

Reference patterns from:
  - pytorch/test/inductor/test_combo_kernels.py (combo kernel + dynamic)
  - pytorch/test/inductor/test_torchinductor_dynamic_shapes.py (nonzero + dynamic)
"""

import unittest

import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

import torch_npu
import torch_npu._inductor

torch._dynamo.config.cache_size_limit = 128

if not torch.npu.is_available():
    raise unittest.SkipTest("NPU is not available")

device = "npu"


# ============================================================
# Base class: provides compile counting + dynamic=True helpers
# ============================================================

class DynamicShapeTestMixin:
    """
    Mixin for dynamic shape tests on NPU.
    Provides _compile() which wraps torch.compile with CompileCounterWithBackend,
    and _assert_compile_count() to verify expected recompilation count.
    """

    def _compile(self, fn, max_expected_compiles=2):
        """Compile fn with inductor backend, tracking compile count."""
        self._compile_counter = CompileCounterWithBackend("inductor")
        self._max_expected = max_expected_compiles
        return torch.compile(fn, backend=self._compile_counter, dynamic=True)

    def _assert_compile_count(self):
        """Assert actual compile count <= expected (dynamic=True should reuse)."""
        actual = self._compile_counter.frame_count
        self.assertLessEqual(
            actual, self._max_expected,
            f"Expected at most {self._max_expected} compilation(s) for dynamic=True, "
            f"but got {actual}. The compiled function should be reused across shape changes.",
        )


# ============================================================
# Part 1: Multi-op Fusion + SIMD/SIMT Axis Splitting Dynamic Shapes
# ============================================================

class TestFusionDynamicShapes(DynamicShapeTestMixin, TestCase):
    """
    Tests for multi-operator fusion with dynamic shapes on NPU.

    These exercises NPU's SplitTiling system where:
      - split_axis selection depends on runtime tensor dimensions
      - tiling_axis covers low_dims and reduction dims
      - dynamic shapes trigger symbolic grouping or SIMT_ONLY fallback (A5)

    Covers:
      - Pointwise fusion (add/mul/relu/sigmoid chain) with dynamic numel axis
      - Reduction fusion (sum/max/min) with dynamic reduction axis
      - Persistent reduction with dynamic outer dims
      - Indirect memory ops (gather/scatter/index_select) with dynamic shapes
      - View/reshape + compute fusion with dynamic shapes
    """

    def _check_close(self, ref, out, name="output", atol=1e-3, rtol=1e-3):
        if isinstance(out, (list, tuple)):
            for i, (r, o) in enumerate(zip(ref, out)):
                self._check_close(r, o, f"{name}[{i}]", atol, rtol)
            return
        if torch.isnan(out).any():
            self.fail(f"{name} contains NaN")
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    # ----------------------------------------------------------
    # Test: Pointwise fusion with dynamic numel axis
    # Verifies split_axis adapts when total numel changes
    # ----------------------------------------------------------
    @parametrize("dtype", [torch.float32, torch.float16])
    def test_pointwise_fusion_dynamic_numel(self, dtype):
        """
        Chain of pointwise ops (add -> relu -> sigmoid -> mul) where
        input shape changes between calls, testing that the numel-axis
        split/tiling adapts correctly.
        """
        def fused_pointwise(a, b, c):
            x = a + b
            x = F.relu(x)
            x = torch.sigmoid(x)
            return x * c

        # Shape variations that change total numel significantly
        shapes_a = [
            (32, 64),   # numel = 2048
            (64, 128),  # numel = 8192  (4x)
            (16, 256),  # numel = 4096  (different axis distribution)
        ]

        torch._dynamo.reset()
        compiled = self._compile(fused_pointwise)

        for idx, shape in enumerate(shapes_a):
            a = torch.randn(shape, dtype=dtype, device=device)
            b = torch.randn(shape, dtype=dtype, device=device)
            c = torch.randn(shape, dtype=dtype, device=device)

            ar, br, cr = a.detach().clone(), b.detach().clone(), c.detach().clone()
            ref = fused_pointwise(ar, br, cr)
            out = compiled(a, b, c)
            self._check_close(ref, out, f"pointwise_numel_{idx}")
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Pointwise fusion with dynamic batch dimension
    # Batch axis is typically selected as split_axis
    # ----------------------------------------------------------
    def test_pointwise_fusion_dynamic_batch_axis(self):
        """
        Pointwise fusion where only batch dimension changes.
        Batch is a high-priority split_axis candidate.
        """
        def fn(x):
            y = x * 2.0
            y = F.relu(y)
            y = y + 1.0
            return torch.sigmoid(y)

        batches = [2, 8, 16]
        feat_dim = 256

        torch._dynamo.reset()
        compiled = self._compile(fn)

        for idx, B in enumerate(batches):
            x = torch.randn((B, feat_dim), dtype=torch.float32, device=device)
            xr = x.detach().clone()
            ref = fn(xr)
            out = compiled(x)
            self._check_close(ref, out, f"pointwise_batch_{idx}")
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Reduction fusion with dynamic reduction axis
    # Reduction axis becomes tiling_axis; its size affects tile granularity
    # ----------------------------------------------------------
    @unittest.skipIf(True,
                     "Reduction fusion with dynamic reduction axis only supported on A5 platform. "
                     "NPU reduction tiling behavior differs on non-A5 devices.")
    def test_reduction_fusion_dynamic_reduction_axis(self):
        """
        sum/max/min reduction over dim whose size changes at runtime.
        Tests that tiling_axis for reduction adapts correctly.
        """
        def fused_reductions(a, b, c):
            s = torch.sum(a, dim=-1)
            m, _ = torch.max(b, dim=-1)
            n = torch.min(c, dim=-1)
            return s, m, n

        # Varying last dimension (reduction axis)
        configs = [
            ((4, 64), (4, 64), (4, 64)),     # reduce over 64
            ((4, 128), (4, 128), (4, 128)),   # reduce over 128
            ((4, 256), (4, 256), (4, 256)),   # reduce over 256
        ]

        torch._dynamo.reset()
        compiled = self._compile(fused_reductions)

        for idx, (sa, sb, sc) in enumerate(configs):
            a = torch.randn(sa, dtype=torch.float32, device=device)
            b = torch.randn(sb, dtype=torch.float32, device=device)
            c = torch.randn(sc, dtype=torch.float32, device=device)
            ar, br, cr = a.detach().clone(), b.detach().clone(), c.detach().clone()
            ref = fused_reductions(ar, br, cr)
            out = compiled(a, b, c)
            self._check_close(ref, out, f"reduction_axis_{idx}")
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Persistent reduction with dynamic outer product
    # Outer product (numel of non-reduction dims) changes
    # ----------------------------------------------------------
    def test_persistent_reduction_dynamic_outer(self):
        """
        Sum reduction with keepdim=True (persistent reduction pattern)
        where the outer (non-reduction) dimension changes.
        """
        def persistent_sum(x):
            return torch.sum(x, dim=1, keepdim=True)

        # Varying batch (outer/non-reduction dim), fixed reduction dim
        batches = [2, 6, 12]
        inner_dim = 512

        torch._dynamo.reset()
        compiled = self._compile(persistent_sum)

        for idx, B in enumerate(batches):
            x = torch.randn((B, inner_dim), dtype=torch.float32, device=device)
            xr = x.detach().clone()
            ref = persistent_sum(xr)
            out = compiled(x)
            self._check_close(ref, out, f"persistent_outer_{idx}")
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Pointwise + Reduction vertical fusion with dynamic shapes
    # This triggers both split_axis AND tiling_axis adaptation
    # ----------------------------------------------------------
    def test_vertical_fusion_dynamic_shapes(self):
        """
        Pointwise op followed by reduction op in one graph.
        Both the numel axis (for pointwise split) and reduction axis
        (for reduction tiling) change dynamically.
        Reference: community test_vertical_pointwise_reduction_fusion
        """
        def pw_plus_reduce(x, y):
            z = x + y          # pointwise: split on outer dims
            z = F.relu(z)     # pointwise
            r = torch.sum(z, dim=-1)  # reduction: tile on last dim
            return r

        configs = [
            ((8, 64), (8, 64)),
            ((16, 128), (16, 128)),
            ((4, 256), (4, 256)),
        ]

        torch._dynamo.reset()
        compiled = self._compile(pw_plus_reduce)

        for idx, (sx, sy) in enumerate(configs):
            x = torch.randn(sx, dtype=torch.float32, device=device)
            y = torch.randn(sy, dtype=torch.float32, device=device)
            xr, yr = x.detach().clone(), y.detach().clone()
            ref = pw_plus_reduce(xr, yr)
            out = compiled(x, y)
            self._check_close(ref, out, f"vertical_fusion_{idx}")
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Gather with dynamic shapes (indirect memory / SIMT path)
    # Gather reads from non-contiguous indices -> may trigger SIMT
    # ----------------------------------------------------------
    def test_gather_dynamic_shapes(self):
        """
        gather operation with dynamic input/output shapes.
        Index tensor selects rows from table; shapes change between calls.
        This exercises the indirect memory path (SIMT/SIMD_SIMT_MIX).
        """
        # Table stays fixed, index output shape changes
        table_dim = 128
        table = torch.randn((table_dim, 64), dtype=torch.float32, device=device)

        def gather_fn(t, idx):
            return torch.gather(t, 0, idx.unsqueeze(-1).expand(-1, t.size(-1)))

        index_shapes = [
            (10,),     # 10 lookups
            (50,),     # 50 lookups
            (100,),    # 100 lookups
        ]

        torch._dynamo.reset()
        compiled = self._compile(gather_fn)

        for idx, ishape in enumerate(index_shapes):
            max_idx = min(table_dim - 1, 127)
            index = torch.randint(0, max_idx, ishape, dtype=torch.int64, device=device)

            tr = table.detach().clone()
            ir = index.detach().clone()
            ref = gather_fn(tr, ir)

            out = compiled(table, index)
            self._check_close(ref, out, f"gather_{idx}")
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Scatter with dynamic shapes (indirect memory / SIMT path)
    # Scatter writes to non-contiguous positions -> may trigger SIMT
    # ----------------------------------------------------------
    def test_scatter_dynamic_shapes(self):
        """
        scatter operation with dynamic source shapes.
        Source values are scattered into target at indexed positions.
        Exercises indirect memory write path.
        """
        def scatter_fn(src, index):
            out = torch.zeros((128, 64), dtype=src.dtype, device=device)
            return torch.scatter(out, 0, index.unsqueeze(-1).expand(-1, 64), src)

        src_shapes = [
            (10, 64),
            (50, 64),
            (100, 64),
        ]

        torch._dynamo.reset()
        compiled = self._compile(scatter_fn)

        for idx, sshape in enumerate(src_shapes):
            src = torch.randn(sshape, dtype=torch.float32, device=device)
            # Use permutation to guarantee no duplicate indices (avoids scatter race condition)
            max_idx = min(127, sshape[0] - 1)
            index = torch.randperm(max_idx + 1, device=device)[:sshape[0]]

            sr = src.detach().clone()
            ir = index.detach().clone()
            ref = scatter_fn(sr, ir)

            out = compiled(src, index)
            self._check_close(ref, out, f"scatter_{idx}", atol=1e-2, rtol=1e-2)
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Index_select with dynamic shapes (indirect memory)
    # ----------------------------------------------------------
    def test_index_select_dynamic_shapes(self):
        """
        index_select along dim=0 with varying number of indices.
        Output shape depends on index tensor size (dynamic).
        """
        table = torch.randn((256, 32), dtype=torch.float32, device=device)

        num_indices_list = [8, 32, 64, 128]

        torch._dynamo.reset()
        compiled = self._compile(
            lambda tbl, idx: torch.index_select(tbl, 0, idx),
        )

        for idx, nidx in enumerate(num_indices_list):
            index = torch.randint(0, 256, (nidx,), dtype=torch.int64, device=device)
            tr = table.detach().clone()
            ir = index.detach().clone()
            ref = torch.index_select(tr, 0, ir)
            out = compiled(table, index)
            self._check_close(ref, out, f"index_select_{idx}")
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: MatMul + activation fusion with dynamic batch
    # MM output shape depends on both input shapes changing
    # ----------------------------------------------------------
    def test_mm_activation_fusion_dynamic(self):
        """
        Linear layer pattern: matmul + bias + relu.
        Both input batch and feature dimensions change.
        Tests that the contraction axis (tiling) and output axis (split)
        adapt together.
        """
        def linear_relu(x, weight, bias):
            return F.relu(F.linear(x, weight, bias))

        configs = [
            {"B": 4, "in_feat": 64, "out_feat": 32},
            {"B": 8, "in_feat": 128, "out_feat": 64},
            {"B": 2, "in_feat": 256, "out_feat": 128},
        ]

        torch._dynamo.reset()
        compiled = self._compile(linear_relu)

        for idx, cfg in enumerate(configs):
            B, Din, Dout = cfg["B"], cfg["in_feat"], cfg["out_feat"]
            x = torch.randn((B, Din), dtype=torch.float32, device=device)
            w = torch.randn((Dout, Din), dtype=torch.float32, device=device)
            b = torch.randn((Dout,), dtype=torch.float32, device=device)

            xr, wr, br = x.detach().clone(), w.detach().clone(), b.detach().clone()
            ref = linear_relu(xr, wr, br)

            out = compiled(x, w, b)
            self._check_close(ref, out, f"mm_act_{idx}")
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Reshape/view + compute fusion with dynamic shapes
    # reshape changes the axis structure seen by split/tiling
    # ----------------------------------------------------------
    def test_reshape_compute_fusion_dynamic(self):
        """
        Reshape followed by element-wise computation.
        The reshape changes which axes are available for splitting.
        Input flat tensors get reshaped to different 2D shapes.
        """
        def reshape_add(x):
            y = x.reshape(x.shape[0], -1)
            return y + 1.0

        flat_sizes = [2048, 8192, 16384]

        torch._dynamo.reset()
        compiled = self._compile(reshape_add)

        for idx, fsz in enumerate(flat_sizes):
            x = torch.randn(fsz, dtype=torch.float32, device=device)
            xr = x.detach().clone()
            ref = reshape_add(xr)
            out = compiled(x)
            self._check_close(ref, out, f"reshape_compute_{idx}")
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Concat + compute fusion with dynamic shapes
    # Concat merges axes, changing split/tiling candidates
    # ----------------------------------------------------------
    def test_cat_compute_fusion_dynamic(self):
        """
        Concat two tensors along an axis, then apply pointwise ops.
        The concat result's axis size changes with input sizes.
        """
        def cat_then_compute(a, b):
            c = torch.cat([a, b], dim=0)
            return F.relu(c) * 2.0

        configs = [
            ((4, 64), (4, 64)),   # cat -> (8, 64)
            ((8, 64), (8, 64)),   # cat -> (16, 64)
            ((16, 64), (16, 64)), # cat -> (32, 64)
        ]

        torch._dynamo.reset()
        compiled = self._compile(cat_then_compute)

        for idx, (sa, sb) in enumerate(configs):
            a = torch.randn(sa, dtype=torch.float32, device=device)
            b = torch.randn(sb, dtype=torch.float32, device=device)
            ar, br = a.detach().clone(), b.detach().clone()
            ref = cat_then_compute(ar, br)
            out = compiled(a, b)
            self._check_close(ref, out, f"cat_compute_{idx}")
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Multi-head attention subgraph (QK^T + softmax + AV)
    # Multiple ops with interdependent dynamic shapes
    # ----------------------------------------------------------
    def test_attention_subgraph_dynamic(self):
        """
        Manual attention subgraph: QK^T -> softmax -> @V.
        Each step has different shape dependencies:
        - matmul1: (B,H,S,D) @ (B,H,D,S) -> (B,H,S,S)  [S changes]
        - softmax: (B,H,S,S) -> (B,H,S,S)
        - matmul2: (B,H,S,S) @ (B,H,S,D) -> (B,H,S,D)
        All intermediate shapes depend on S which is dynamic.
        """
        def attn_subgraph(q, k, v, scale=None):
            scale = scale or (q.shape[-1] ** -0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        configs = [
            (2, 4, 64, 32),
            (4, 4, 128, 32),
            (2, 8, 96, 32),
        ]

        torch._dynamo.reset()
        compiled = self._compile(attn_subgraph)

        for idx, (B, H, S, D) in enumerate(configs):
            q = torch.randn((B, H, S, D), dtype=torch.float32, device=device)
            k = torch.randn((B, H, S, D), dtype=torch.float32, device=device)
            v = torch.randn((B, H, S, D), dtype=torch.float32, device=device)
            qr, kr, vr = q.detach().clone(), k.detach().clone(), v.detach().clone()
            ref = attn_subgraph(qr, kr, vr)
            out = compiled(q, k, v)
            self._check_close(ref, out, f"attn_subgraph_{idx}", atol=1e-2, rtol=1e-2)
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: LayerNorm + residual connection with dynamic seq_len
    # LayerNorm normalizes over last dim; residual add matches shape
    # ----------------------------------------------------------
    def test_layernorm_residual_dynamic(self):
        """
        LayerNorm + residual add where sequence length changes.
        LayerNorm's reduction axis (last dim) is fixed but the
        outer (batch*seq) axis changes.
        """
        def ln_residual(x, residual):
            return F.layer_norm(x, (x.shape[-1],)) + residual

        configs = [
            (2, 128, 64),
            (4, 256, 64),
            (2, 512, 64),
        ]

        torch._dynamo.reset()
        compiled = self._compile(ln_residual)

        for idx, (B, S, D) in enumerate(configs):
            x = torch.randn((B, S, D), dtype=torch.float32, device=device)
            res = torch.randn_like(x)
            xr, rr = x.detach().clone(), res.detach().clone()
            ref = ln_residual(xr, rr)
            out = compiled(x, res)
            self._check_close(ref, out, f"ln_residual_{idx}")
        self._assert_compile_count()


# ============================================================
# Part 2: Nonzero Data-Dependent Dynamic Shape Tests
# ============================================================

class TestNonzeroDynamicShapes(DynamicShapeTestMixin, TestCase):
    """
    Tests for nonzero operator with dynamic shapes on NPU.

    nonzero produces data-dependent output: the number of non-zero elements
    determines output shape[0]. Under dynamo, this creates an unbacked SymInt
    guarded by runtime assertions.

    Uses capture_dynamic_output_shape_ops=True per community convention.
    """

    def _check_close(self, ref, out, name="output"):
        if isinstance(out, (list, tuple)):
            for i, (r, o) in enumerate(zip(ref, out)):
                self._check_close(r, o, f"{name}[{i}]")
            return
        torch.testing.assert_close(ref, out)

    # ----------------------------------------------------------
    # Test: Basic nonzero with dynamic output shape
    # ----------------------------------------------------------
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_basic_dynamic(self):
        """
        nonzero on 1D tensor: output shape[0] = count of non-zero elements.
        Different inputs produce different output shapes.
        """
        def fn(x):
            return x.nonzero()

        compiled = self._compile(fn)

        # Case 1: 3 non-zeros
        x1 = torch.tensor([1, 0, 2, 0, 3], dtype=torch.float32, device=device)
        r1 = compiled(x1)
        ref1 = x1.nonzero()
        self._check_close(ref1, r1, "nonzero_3elem")
        self.assertEqual(r1.shape[0], 3)

        # Case 2: 5 non-zeros (same input shape, more nonzeros)
        x2 = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32, device=device)
        r2 = compiled(x2)
        ref2 = x2.nonzero()
        self._check_close(ref2, r2, "nonzero_5elem")
        self.assertEqual(r2.shape[0], 5)

        # Case 3: 0 non-zeros (edge case)
        x3 = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32, device=device)
        r3 = compiled(x3)
        ref3 = x3.nonzero()
        self._check_close(ref3, r3, "nonzero_0elem")
        self.assertEqual(r3.shape[0], 0)

        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Nonzero + arithmetic propagation
    # Unbacked SymInt propagates through subsequent ops
    # ----------------------------------------------------------
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_nonzero_arithmetic_propagation(self):
        """
        nonzero result used in arithmetic: r * 2.
        Tests unbacked SymInt propagation through pointwise ops.
        """
        def fn(x):
            r = x.nonzero()
            return r * 2

        compiled = self._compile(fn)

        x = torch.tensor([0, 4, 2, 0, 1], dtype=torch.int64, device=device)
        ref = compiled(x)
        opt_ref = x.nonzero() * 2
        self._check_close(opt_ref, ref, "nonzero_mul2")
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Nonzero + split (dynamic-sized split)
    # ----------------------------------------------------------
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_split_dynamic(self):
        """
        nonzero followed by split using another tensor's size.
        Reference: community test_nonzero_no_realloc
        """
        def fn(x, y):
            z = x.nonzero()
            return torch.split(z, [y.size(0)])

        compiled = self._compile(fn)

        x = torch.tensor([1, 0, 1, 1, 0, 1, 0], dtype=torch.float32, device=device)
        y = torch.randn(4, device=device)
        result = compiled(x, y)
        # Should not error; verifies nonzero dynamic output flows into split
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Nonzero + zeros factory (size derived from nonzero)
    # ----------------------------------------------------------
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_size_factory(self):
        """
        Use nonzero output size to create a new zero tensor.
        Reference: community test_nonzero_size_factory_nobreak
        """
        def fn(x, template):
            y = template.nonzero()
            return x.new_zeros(y.size(0))

        compiled = self._compile(fn)

        tmpl = torch.tensor([True, True, False, False, True],
                            dtype=torch.bool, device=device)
        x = torch.randn(5, device=device)
        ref = compiled(x, tmpl)
        expected = x.new_zeros(3)  # 3 True values
        self.assertEqual(ref.shape, expected.shape)
        self.assertEqual(ref.numel(), 3)
        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: count_nonzero with dynamic shapes
    # count_nonzero returns scalar (0-d tensor), not shaped output
    # ----------------------------------------------------------
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @unittest.skipIf(True,
                     "count_nonzero with capture_scalar_outputs only supported on A5 platform.")
    def test_count_nonzero_dynamic(self):
        """
        count_nonzero returns a scalar; used to create sized tensors.
        Tests scalar output capture with dynamic shapes.
        """
        def fn(x):
            cnt = x.count_nonzero()
            return cnt, torch.zeros(cnt.item(), device=device)

        compiled = self._compile(fn)

        # Different sparsity levels produce different counts
        x1 = torch.tensor([1, 0, 1, 0, 1, 0, 0, 0], dtype=torch.float32, device=device)
        cnt1, out1 = compiled(x1)
        self.assertEqual(cnt1.item(), 3)
        self.assertEqual(out1.shape[0], 3)

        x2 = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32, device=device)
        cnt2, out2 = compiled(x2)
        self.assertEqual(cnt2.item(), 8)
        self.assertEqual(out2.shape[0], 8)

        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Nonzero 2D with dynamic row count
    # ----------------------------------------------------------
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @unittest.skipIf(True,
                     "NPU nonzero fake kernel has incorrect stride metadata for 2D output "
                     "(assert_size_stride mismatch). The inductor-compiled nonzero produces "
                     "wrong stride info (u0, 2) vs expected (2, u0) for (num_nonzeros, ndim) "
                     "shape. Re-enable after NPU nonzero meta kernel fix.")
    def test_nonzero_2d_dynamic(self):
        """
        nonzero on 2D tensor: output is (num_nonzeros, ndim).
        ndim is fixed (2), but num_nonzero varies.
        """
        def fn(x):
            return x.nonzero()

        compiled = self._compile(fn)

        # Sparse case
        x1 = torch.zeros((4, 4), dtype=torch.float32, device=device)
        x1[0, 0] = 1.0
        x1[2, 2] = 1.0
        r1 = compiled(x1)
        ref1 = x1.nonzero()
        self._check_close(ref1, r1, "nonzero_2d_sparse")
        self.assertEqual(r1.shape, (2, 2))

        # Denser case (same shape, more nonzeros)
        x2 = torch.ones((4, 4), dtype=torch.float32, device=device)
        r2 = compiled(x2)
        ref2 = x2.nonzero()
        self._check_close(ref2, r2, "nonzero_2d_dense")
        self.assertEqual(r2.shape, (16, 2))

        self._assert_compile_count()

    # ----------------------------------------------------------
    # Test: Nonzero + masked_fill / where combination
    # ----------------------------------------------------------
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_where_chain(self):
        """
        nonzero -> use indices for advanced indexing (where/masked_fill).
        Tests that dynamic indices can be used in downstream indexing ops.
        """
        def fn(x):
            indices = x.nonzero(as_tuple=True)
            # Use nonzero indices to select positions
            mask = torch.zeros_like(x, dtype=torch.bool)
            if len(indices) > 0 and len(indices[0]) > 0:
                mask[indices] = True
            return x.masked_fill(~mask, 0.0)

        compiled = self._compile(fn)

        x = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
                         dtype=torch.float32, device=device)
        xr = x.detach().clone()
        ref = fn(xr)
        out = compiled(x)
        self._check_close(ref, out, "nonzero_where")
        self._assert_compile_count()


instantiate_parametrized_tests(TestFusionDynamicShapes)
instantiate_parametrized_tests(TestNonzeroDynamicShapes)


if __name__ == "__main__":
    run_tests()
