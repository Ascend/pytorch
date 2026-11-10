import os
import re
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._inductor.exc import InductorError
from torch._inductor.utils import IndentedBuffer, run_and_get_code
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests

import torch_npu
import torch_npu.testing  # noqa: F401


class TestReductionShapeAssertions(TestCase):
    def test_deferred_reduction_shape(self):
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonKernel

        kernel = SimpleNamespace(
            _npu_linearize=True,
            features=SimpleNamespace(is_reduction=lambda: True),
            _npu_assert_shapes={},
        )
        buffer = IndentedBuffer()
        buffer.writeline("tl.static_assert(tmp0.dtype == tl.float32)")
        NPUTritonKernel.codegen_shape_assert(kernel, buffer, "tmp0", ("R0_BLOCK",))
        self.assertIn("tmp0.shape[0] == R0_BLOCK", buffer.getvalue())
        kernel._npu_assert_shapes["tmp0"] = (1, "R0_BLOCK")
        source = buffer.getvalue()
        self.assertIn("len(tmp0.shape) == 2", source)
        self.assertIn("tmp0.shape[0] == 1", source)
        self.assertIn("tmp0.shape[1] == R0_BLOCK", source)
        self.assertIn("tl.static_assert(tmp0.dtype == tl.float32)", source)


class TestNarrowLeafMask(TestCase):
    """Pure-code: narrowing a load's mask must never change its semantics."""

    def narrow(self, source, drop):
        import ast

        from torch_npu._inductor.triton_experimental.codegen.triton import _npu_narrow_leaf_mask

        result = _npu_narrow_leaf_mask(ast.parse(source, mode="eval").body, set(drop))
        return None if result is None else ast.unparse(result)

    def test_narrows_and_chain(self):
        self.assertEqual(self.narrow("r0_1mask & r0_0mask", {"r0_1mask"}), "r0_0mask")

    def test_keeps_non_leaf_atoms(self):
        self.assertEqual(
            self.narrow("r0_1mask & r0_0mask & xmask", {"r0_1mask"}), "r0_0mask & xmask"
        )
        self.assertEqual(
            self.narrow("xmask & r0_1mask & r0_0mask", {"r0_1mask"}), "xmask & r0_0mask"
        )

    def test_refuses_non_and_structure(self):
        # "|", "~", comparisons, literals and calls are all reasons to leave the
        # mask alone rather than rebuild it as an "&" chain of names.
        for source in (
            "r0_1mask | r0_0mask",
            "~r0_1mask & r0_0mask",
            "r0_1mask & r0_0mask | xmask",
            "r0_0mask & (x0 < 4)",
            "r0_1mask & r0_0mask & True",
            "r0_1mask & tl.full([], 1, tl.int1)",
        ):
            self.assertIsNone(self.narrow(source, {"r0_1mask"}), source)

    def test_refuses_empty_and_noop(self):
        self.assertIsNone(self.narrow("r0_1mask", {"r0_1mask"}))
        self.assertIsNone(self.narrow("r0_1mask & r0_1mask", {"r0_1mask"}))
        self.assertIsNone(self.narrow("r0_0mask & xmask", {"r0_1mask"}))


class TestReductionShape(TestCase):
    @dtypes(torch.float16, torch.float32)
    @parametrize("length", [20, 257])
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_dot(self, device, dtype, length):
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonKernel

        original_codegen = NPUTritonKernel.codegen_kernel
        checked_updates = []

        def codegen(kernel, *args, **kwargs):
            source = original_codegen(kernel, *args, **kwargs)
            for name, var in kernel.cse.varname_map.items():
                op = getattr(var, "_npu_shape_op", None)
                if op is None or op[0] != "add" or len(op[1]) != 2:
                    continue
                if not all(getattr(arg, "_npu_reduction_shape", None) is not None for arg in op[1]):
                    continue
                for value in (*op[1], var):
                    self.assertEqual(kernel._npu_assert_shapes[value.name], ("R0_BLOCK", 1))
                self.assertRegex(source, rf"tl\.static_assert\([^\n]*{name}\.shape\[0\] == R0_BLOCK")
                self.assertRegex(source, rf"tl\.static_assert\([^\n]*{name}\.shape\[1\] == 1")
                checked_updates.append(name)
            return source

        x = torch.linspace(-0.5, 0.5, length, device=device, dtype=dtype).requires_grad_()
        y = torch.linspace(0.25, 0.75, length, device=device, dtype=dtype).requires_grad_()

        def fn(x, y):
            return (x * y).sum()

        compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})
        expected = fn(x, y)
        with mock.patch.object(NPUTritonKernel, "codegen_kernel", codegen):
            actual, codes = run_and_get_code(compiled, x, y)
        self.assertTrue(checked_updates)
        self.assertEqual(actual, expected)
        self.assertEqual(torch.autograd.grad(actual, (x, y)), torch.autograd.grad(expected, (x, y)))
        self.assertRegex("\n".join(codes), r"tl\.static_assert\(.*\.shape\[0\] == R0_BLOCK")

    @dtypes(torch.float32)
    @parametrize("axis", [0, 1])
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_axis(self, device, dtype, axis):
        shape = (257, 20) if axis == 0 else (5, 257)
        x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

        def fn(x):
            return x.square().sum(dim=axis)

        compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})
        actual, codes = run_and_get_code(compiled, x)
        expected = fn(x)
        self.assertEqual(actual, expected)
        self.assertEqual(torch.autograd.grad(actual.sum(), x), torch.autograd.grad(expected.sum(), x))
        source = "\n".join(codes)
        self.assertRegex(source, r"tl\.static_assert\(.*\.shape\[\d+\] == R0_BLOCK")
        pointwise = re.search(r"tl\.arange\(0, (real_block_\w+)\)", source)
        self.assertIsNotNone(pointwise)
        self.assertRegex(source, rf"tl\.static_assert\([^\n]*\.shape\[\d+\] == {re.escape(pointwise.group(1))}")
        self.assertNotRegex(source, r"tl\.static_assert\(\w+\.shape ==")

    @dtypes(torch.float16, torch.float32)
    @parametrize("shape", [(3, 4), (5, 7), (13, 17)])
    @parametrize("case", ["row", "col", "both", "scalar_on_row", "scalar_on_col"])
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
        "implicit_fallbacks": False,
    })
    def test_partial_axis_broadcast(self, device, dtype, case, shape):
        # A promoted r-tree gives each free r-node its own register axis, so an
        # operand that spans only some of them is loaded narrow and broadcast to
        # the tile. Doing elementwise work on such an operand used to emit a load
        # whose index was broadcast instead of its value, which compiles but keeps
        # a bar shape in HIR: the next op fails with "'hivm.hir.vadd' op operands'
        # shape are inconsistent". The scalar_on_* cases are that regression.
        # The non-divisible shapes are the point: dropping an uncovered axis's
        # mask must stay correct in the tail tile, where those lanes read real
        # data instead of ``other`` and only the reduction's own mask discards
        # them. The load's address never depends on a dropped axis, so no lane
        # reads out of bounds.
        rows, cols = shape
        x = torch.linspace(-3, 3, rows * cols, device=device, dtype=dtype).reshape(rows, cols)
        row = torch.linspace(0.5, 2.0, cols, device=device, dtype=dtype)
        col = torch.linspace(0.75, 1.25, rows, device=device, dtype=dtype).reshape(rows, 1)

        if case == "row":
            def fn(a):
                return (a * row).mean()
        elif case == "col":
            def fn(a):
                return (a * col).mean()
        elif case == "both":
            def fn(a):
                return (a * row * col).mean()
        elif case == "scalar_on_row":
            def fn(a):
                return ((row - 1.0) * a * col).mean()
        else:
            def fn(a):
                return ((col - 1.0) * a * row).mean()

        compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})
        expected = fn(x)
        actual, codes = run_and_get_code(compiled, x)
        atol, rtol = (1e-3, 1e-3) if dtype == torch.float16 else (1e-5, 1.3e-5)
        self.assertEqual(actual, expected, atol=atol, rtol=rtol)
        # The narrow load must broadcast its value, never its index.
        source = "\n".join(codes)
        self.assertNotRegex(source, r"tl\.load\([^\n]*tl\.broadcast_to\(")

    @dtypes(torch.float32)
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_unknown_load(self, device, dtype):
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonKernel

        original = NPUTritonKernel.load
        original_codegen = NPUTritonKernel.codegen_kernel
        unknown_names = set()
        captured = []

        class SourceCaptured(RuntimeError):
            pass

        def load(kernel, *args, **kwargs):
            result = original(kernel, *args, **kwargs)
            result._npu_load_index = None
            unknown_names.add(result.name)
            return result

        def codegen(kernel, *args, **kwargs):
            consumers = {}
            for name, var in kernel.cse.varname_map.items():
                op = getattr(var, "_npu_shape_op", None)
                if op is not None and any(getattr(arg, "name", None) in unknown_names for arg in op[1]):
                    unknown_names.add(name)
                    consumers[name] = var.shape
            source = original_codegen(kernel, *args, **kwargs)
            captured.append((source, consumers))
            raise SourceCaptured("unknown-load source captured before compilation")

        x = torch.randn(20, device=device, dtype=dtype)

        def fn(x):
            return (x * x).sum()

        with (
            mock.patch.object(NPUTritonKernel, "load", load),
            mock.patch.object(NPUTritonKernel, "codegen_kernel", codegen),
        ):
            compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})
            with self.assertRaisesRegex(InductorError, "unknown-load source captured before compilation"):
                compiled(x)
        self.assertEqual(len(captured), 1)
        source, consumers = captured[0]
        self.assertTrue(consumers)
        self.assertRegex(source, r"\w+ = tl\.load\(")
        for name, shape in consumers.items():
            checks = [f"len({name}.shape) == {len(shape)}"]
            checks.extend(f"{name}.shape[{axis}] == {dim}" for axis, dim in enumerate(shape))
            for check in checks:
                self.assertRegex(source, rf"tl\.static_assert\([^\n]*{re.escape(check)}")

    @dtypes(torch.float32)
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_without_linearize(self, device, dtype):
        from torch_npu._inductor.triton_experimental.codegen import triton as codegen

        x = torch.randn(257, device=device, dtype=dtype)

        def fn(x):
            return x.square() + 1

        with mock.patch.object(codegen, "triton_codegen_linearize", False):
            compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})
            actual, codes = run_and_get_code(compiled, x)
        self.assertEqual(actual, fn(x))
        source = "\n".join(codes)
        self.assertRegex(source, r"tl\.static_assert\(.*\.shape\[0\] == XBLOCK")
        self.assertNotRegex(source, r"tl\.static_assert\(\w+\.shape ==")


def _compile_fwd_bwd(fn, args, wrt):
    """Compile fn, run forward and backward, and return the generated source.

    `wrt` selects the inputs to differentiate, which need not be all args.

    Module-level on purpose, NOT a @staticmethod on the TestCase:
    instantiate_device_type_tests ports non-test members with
    `getattr(generic_cls, name)` + `setattr(device_cls, name, ...)`, which
    unwraps a staticmethod into a plain function and rebinds it as an instance
    method on the generated class, so `self._helper(a, b, c)` would pass self
    as a fourth argument (TypeError observed on the CI interpreter).
    """
    from torch._inductor.graph import GraphLowering

    codes = []
    with mock.patch.object(GraphLowering, "save_output_code", side_effect=codes.append):
        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph=True,
                                 options={"npu_backend": "triton_experimental"})
        out = compiled(*args)
        grads = torch.autograd.grad(out.sum(), wrt)
    return out, grads, "\n".join(codes)


def _split_kernels(source):
    """[(kernel_name, body)] -- tmpN names are only unique within a kernel."""
    starts = [(m.start(), m.group(1))
              for m in re.finditer(r"^def (triton_\w+)\(", source, re.M)]
    out = []
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(source)
        out.append((name, source[pos:end]))
    return out


class TestScalarBlockLoadShape(TestCase):
    """Guards for the constant-index scalar-load provenance, both directions.

    A 0-dim operand is loaded at a constant index and, in a NON-degenerate
    kernel, upstream's load() reassigns result_var to
    `tl.broadcast_to(tmp, expand_shape)`, so the value it returns -- and every
    consumer, e.g. `.to(tl.float32)` -- has rank >= 1 while reconstructing the
    shape from the index alone yields (). Asserting `len(var.shape) == 0`
    against that value fails to compile, so the reconstruction keeps upstream's
    expected shape for exactly this provenance.

    In a DEGENERATE kernel the [1] broadcast is stripped from the final source,
    so () stays correct there and rank-0 assertions must still be emitted.
    """

    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "implicit_fallbacks": False,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_scalar_cast_consumer_not_asserted_rank0(self, device):
        """Non-degenerate kernel: a broadcast scalar load feeding a dtype cast
        must be asserted at its real rank, never rank 0.

        The cast is created EXPLICITLY (a float16 0-dim operand converted to
        float32) rather than left to dtype promotion: a same-dtype graph emits
        no convert node at all, which would make this guard vacuous. This is
        the provenance KLDiv's fp16 backward hit, reproduced without KLDiv.
        """
        scalar = torch.tensor(2.5, device=device, dtype=torch.float16)
        vec = torch.linspace(-1.0, 1.0, 8, device=device, dtype=torch.float32).requires_grad_()

        def fn(v, s):
            return (v * s.to(torch.float32)).sum()

        expected = fn(vec, scalar)
        expected_grads = torch.autograd.grad(expected, (vec,))
        actual, actual_grads, source = _compile_fwd_bwd(fn, (vec, scalar), (vec,))
        self.assertEqual(actual, expected)
        self.assertEqual(actual_grads, expected_grads, atol=0, rtol=0)

        # tmpN names are per-kernel, and `source` concatenates the forward and
        # backward wrappers, so every check below is scoped to ONE kernel body.
        # Without that scoping a backward broadcast (e.g. tmp3) collides with a
        # legitimately rank-0 tmp3 in the forward kernel.
        kernels = _split_kernels(source)
        self.assertTrue(kernels, f"no triton kernels in generated source:\n{source[:3000]}")
        checked = 0
        found_broadcast = False
        for kname, body in kernels:
            broadcasts = re.findall(r"(tmp\d+) = tl\.broadcast_to\((tmp\d+), \[[^\]]+\]\)", body)
            if broadcasts:
                found_broadcast = True
            for bcast_var, _loaded in broadcasts:
                self.assertNotIn(
                    f"tl.static_assert(len({bcast_var}.shape) == 0)", body,
                    f"{kname}: broadcast {bcast_var} asserted rank 0",
                )
                for consumer in re.findall(rf"(tmp\d+) = {re.escape(bcast_var)}\.to\(", body):
                    self.assertNotIn(
                        f"tl.static_assert(len({consumer}.shape) == 0)", body,
                        f"{kname}: cast consumer {consumer} of {bcast_var} asserted rank 0",
                    )
                    self.assertRegex(
                        body,
                        rf"tl\.static_assert\(\(?len\({consumer}\.shape\) == [1-9]",
                        f"{kname}: cast consumer {consumer} of {bcast_var} carries no "
                        f"rank>=1 shape assertion:\n{body[:3000]}",
                    )
                    checked += 1
        self.assertTrue(
            found_broadcast,
            f"expected a broadcast scalar load in some kernel:\n{source[:3000]}",
        )
        self.assertTrue(
            checked,
            f"no dtype-cast consumer of a broadcast scalar load was generated in any "
            f"kernel; the guard would be vacuous:\n{source[:3000]}",
        )

    @dtypes(torch.float32)
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "implicit_fallbacks": False,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_degenerate_kernel_still_asserts_rank0(self, device, dtype):
        """Degenerate kernel: rank-0 reconstruction stays in force.

        Every non-reduction range tree has numel 1, so the [1] broadcast is
        stripped and scalar values really are rank 0; at least one rank-0 shape
        assertion must still be emitted.
        """
        scalar = torch.tensor(2.5, device=device, dtype=dtype, requires_grad=True)
        one = torch.tensor([3.0], device=device, dtype=dtype, requires_grad=True)

        def fn(s, v):
            return v * s

        expected = fn(scalar, one)
        expected_grads = torch.autograd.grad(expected.sum(), (scalar, one))
        args = (scalar, one)
        actual, actual_grads, source = _compile_fwd_bwd(fn, args, args)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_grads, expected_grads, atol=0, rtol=0)
        self.assertRegex(
            source,
            r"tl\.static_assert\(len\(tmp\d+\.shape\) == 0\)",
            f"degenerate kernel emitted no rank-0 shape assertion:\n{source[:3000]}",
        )


instantiate_device_type_tests(TestScalarBlockLoadShape, globals(), only_for="npu")

instantiate_device_type_tests(TestReductionShape, globals(), only_for="npu")

if __name__ == "__main__":
    run_tests()
