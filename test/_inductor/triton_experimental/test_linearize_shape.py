import contextlib
import os
import re
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._inductor.utils import IndentedBuffer, run_and_get_code
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    TestCase, instantiate_parametrized_tests, parametrize, run_tests,
)

import torch_npu
import torch_npu.testing  # noqa: F401


class TestDeferredShapeAssertions(TestCase):
    def test_shape_state_survives_splice(self):
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonKernel

        def make_kernel():
            return SimpleNamespace(
                _npu_linearize=True,
                features=SimpleNamespace(is_reduction=lambda: False),
                _npu_assert_shapes={},
            )

        first, second = make_kernel(), make_kernel()
        source = IndentedBuffer()
        with source.indent():
            NPUTritonKernel.codegen_shape_assert(first, source, "tmp0", ("XBLOCK",))
        # Rendering early must not freeze the expected shape.
        self.assertIn("tmp0.shape[0] == XBLOCK", source.getvalue())
        output = IndentedBuffer()
        with output.indent():
            output.splice(source)
        first._npu_assert_shapes["tmp0"] = ("real_block_x1", 1)
        actual = output.getvalue()
        self.assertTrue(actual.startswith("    tl.static_assert("))
        self.assertIn("len(tmp0.shape) == 2", actual)
        self.assertIn("tmp0.shape[0] == real_block_x1", actual)
        self.assertIn("tmp0.shape[1] == 1", actual)
        self.assertNotIn("XBLOCK", actual)

        other = IndentedBuffer()
        NPUTritonKernel.codegen_shape_assert(second, other, "tmp0", ())
        self.assertEqual(other.getvalue(), "tl.static_assert(len(tmp0.shape) == 0)\n")
        first._npu_assert_shapes["tmp0"] = None
        self.assertIn("tmp0.shape[0] == XBLOCK", output.getvalue())


@instantiate_parametrized_tests
class TestMaskedShapeMetadata(TestCase):
    @parametrize("producer", ["add", "where"])
    @torch._inductor.config.patch({"test_configs.runtime_triton_shape_assert": True})
    def test_preserve_producer(self, producer):
        from torch.utils._sympy.value_ranges import ValueRanges
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonCSEVariable

        result = NPUTritonCSEVariable("tmp0", ValueRanges.unknown(), torch.float32, ("XBLOCK",))
        result.update_on_args(producer, (1, 2), {})
        original = result._npu_shape_op
        result.update_on_args("masked", (object(), object(), 0), {})
        self.assertIs(result._npu_shape_op, original)

    @parametrize("target", ["load", "add", "fused_load", "load_where"])
    @parametrize("reuse", [False, True])
    @torch._inductor.config.patch({"test_configs.runtime_triton_shape_assert": True})
    def test_masked_fill_source(self, target, reuse):
        from torch._inductor.shape_propagation import ShapePropagationOpsHandler
        from torch._inductor.virtualized import V
        from torch.utils._sympy.value_ranges import ValueRanges
        from torch_npu._inductor.triton_experimental.codegen.triton import (
            NPUTritonCSEVariable, NPUTritonKernelOverrides,
        )

        def var(name):
            return NPUTritonCSEVariable(name, ValueRanges.unknown(), torch.float32, ("XBLOCK",))

        result, fill, selected = var("tmp0"), var("tmp1"), var("tmp2")
        result.update_on_args("add", (1, 2), {})
        if target == "load_where":
            result._npu_shape_op = ("where", (1, fill, fill), {})
        original_fill = ("identity", (result,), {}) if reuse else None
        fill._npu_shape_op = original_fill
        node = SimpleNamespace(target="add" if target == "add" else "load", args=("buffer", "input0"))
        output = SimpleNamespace(args=(node,))
        body = mock.Mock(return_value=result, graph=mock.Mock(find_nodes=mock.Mock(return_value=[output])))
        cse = SimpleNamespace(generate=mock.Mock(return_value=fill), store_cache={})
        if target == "fused_load":
            cse.store_cache["input0"] = result
        kernel = SimpleNamespace(
            cse=cse, compute=IndentedBuffer(), mask_loads=mock.Mock(return_value=contextlib.nullcontext("mask")),
        )

        def select(mask, value, other):
            selected.update_on_args("where", (mask, value, other), {})
            return selected

        where = mock.Mock(side_effect=select)
        with (
            V.set_kernel_handler(kernel),
            V.set_ops_handler(SimpleNamespace(where=where)),
            V.set_graph_handler(SimpleNamespace(is_unspec_arg=lambda _: False)),
        ):
            actual = NPUTritonKernelOverrides.masked("mask", body, 0)
        body.assert_called_once_with()
        if target in ("load", "load_where"):
            self.assertIs(actual, result)
            cse.generate.assert_not_called()
            where.assert_not_called()
            self.assertIs(fill._npu_shape_op, original_fill)
        else:
            self.assertIs(actual, selected)
            where.assert_called_once_with("mask", result, fill)
            self.assertEqual(fill._npu_shape_op, ("identity", (result,), {}))
            if reuse:
                self.assertIs(fill._npu_shape_op, original_fill)
            # The dependency is evaluated with the final shape, including unknown.
            handler = ShapePropagationOpsHandler()
            self.assertEqual(handler.identity(SimpleNamespace(shape=("real_block_x0", 1))), ("real_block_x0", 1))
            self.assertIsNone(handler.identity(SimpleNamespace(shape=None)))


    @torch._inductor.config.patch({"test_configs.runtime_triton_shape_assert": False})
    def test_disabled_delegates_to_upstream(self):
        from torch._inductor.codegen.triton import TritonKernelOverrides
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonKernelOverrides

        body = object()
        with mock.patch.object(TritonKernelOverrides, "masked", return_value=mock.sentinel.result) as parent:
            result = NPUTritonKernelOverrides.masked("mask", body, 0)
        self.assertIs(result, mock.sentinel.result)
        parent.assert_called_once_with("mask", body, 0)


class TestLinearizeShape(TestCase):
    @dtypes(torch.float16, torch.float32)
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_partial_tile(self, device, dtype):
        torch._dynamo.reset()
        x = torch.randn((5, 5, 5), device=device, dtype=dtype, requires_grad=True)
        mask = torch.randint(0, 2, (5, 5, 1), device=device).to(dtype)

        def fn(x, mask):
            return x * (mask * 2)

        compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})
        expected = fn(x, mask)
        actual, codes = run_and_get_code(compiled, x, mask)
        self.assertEqual(actual, expected)
        self.assertEqual(torch.autograd.grad(actual.sum(), x), torch.autograd.grad(expected.sum(), x))
        self.assertRegex(
            "\n".join(codes),
            r"len\((\w+)\.shape\) == 2\) and \(\1\.shape\[0\] == real_block_\w+\)\) and \(\1\.shape\[1\] == 1\)",
        )

    @dtypes(torch.float16, torch.float32)
    @parametrize("operation", ["pad", "cat"])
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "force_pointwise_cat": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_masked_partial_tile(self, device, dtype, operation):
        from torch._inductor import lowering
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonKernelOverrides

        original_masked = NPUTritonKernelOverrides.masked
        fills = []

        def masked(mask, body, other):
            result = original_masked(mask, body, other)
            op = getattr(result, "_npu_shape_op", None)
            if op is not None and op[0] == "where":
                fill = op[1][2]
                source = getattr(fill, "_npu_shape_op", None)
                if source is not None and source[0] == "identity":
                    fills.append(fill)
            return result

        x = torch.randn((5, 5, 5), device=device, dtype=dtype, requires_grad=True)

        def fn(x):
            value = x.square() + 1
            if operation == "pad":
                return torch.nn.functional.pad(value, (1, 2))
            return torch.cat((value, value + 2), dim=2)

        torch._dynamo.reset()
        # Exercise TE codegen even on devices whose default lowering is a fallback.
        with (
            mock.patch.dict(lowering.lowerings, {
                torch.ops.aten.cat.default: lowering.cat,
                torch.ops.aten.constant_pad_nd.default: lowering.constant_pad_nd,
            }),
            mock.patch.object(NPUTritonKernelOverrides, "masked", staticmethod(masked)),
        ):
            actual, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"}), x
            )
        expected = fn(x)
        self.assertEqual(actual, expected)
        self.assertEqual(torch.autograd.grad(actual.sum(), x), torch.autograd.grad(expected.sum(), x))
        self.assertTrue(fills)
        source = "\n".join(codes)
        self.assertIn("tl.where", source)
        self.assertRegex(source, r"real_block_\w+")
        self.assertNotRegex(source, r"tl\.static_assert\(\w+\.shape ==")

    @dtypes(torch.float32)
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_non_linearize_shape_assert(self, device, dtype):
        from torch_npu._inductor.triton_experimental.codegen import triton as codegen

        x = torch.randn((128,), device=device, dtype=dtype)

        def fn(x):
            return x.sin() + 1

        torch._dynamo.reset()
        with mock.patch.object(codegen, "triton_codegen_linearize", False):
            actual, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"}), x
            )
        self.assertEqual(actual, fn(x))
        self.assertRegex("\n".join(codes), r"len\(\w+\.shape\) == 1\) and \(\w+\.shape\[0\] == XBLOCK")

    @dtypes(torch.float32)
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_scalar_load(self, device, dtype):
        torch._dynamo.reset()
        x = torch.randn((), device=device, dtype=dtype, requires_grad=True)
        weight = torch.tensor(0.5, device=device, dtype=dtype, requires_grad=True)

        def fn(x, weight):
            return x * weight

        compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})
        expected = fn(x, weight)
        actual, codes = run_and_get_code(compiled, x, weight)
        self.assertEqual(actual, expected)
        self.assertEqual(torch.autograd.grad(actual, (x, weight)), torch.autograd.grad(expected, (x, weight)))
        self.assertRegex("\n".join(codes), r"len\(\w+\.shape\) == 0")

    @dtypes(torch.float32)
    @parametrize("shape_assert", [False, True])
    def test_arange_shape_assert(self, device, dtype, shape_assert):
        x = torch.randn((5, 25), device=device, dtype=dtype)

        def fn(x):
            offsets = torch.arange(x.shape[-1], device=x.device, dtype=torch.int64)
            return (x + offsets) * 2

        torch._dynamo.reset()
        with torch._inductor.config.patch({
            "force_disable_caches": True,
            "test_configs.runtime_triton_shape_assert": shape_assert,
            "test_configs.runtime_triton_dtype_assert": True,
        }):
            actual, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"}), x
            )
        self.assertEqual(actual, fn(x))
        assertions = re.findall(r"tl\.static_assert\([^\n]*\.shape[^\n]*", "\n".join(codes))
        if shape_assert:
            self.assertTrue(assertions)
            self.assertNotIn("XBLOCK", "\n".join(assertions))
        else:
            self.assertEqual(assertions, [])


    @dtypes(torch.float32)
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_value_expr_after_upstream_conversion(self, device, dtype):
        from torch._inductor import optimize_indexing
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonCSEVariable

        x = torch.randn((5, 25), device=device, dtype=dtype)

        def fn(x):
            return x + torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)

        original_reduce = optimize_indexing.indexing_dtype_strength_reduction
        original_update = NPUTritonCSEVariable.update_on_args
        seen = set()

        def prepare(body):
            original_reduce(body)
            optimize_indexing.convert_index_expr_to_value_expr(body)

        def update(var, name, args, kwargs):
            seen.add(name)
            return original_update(var, name, args, kwargs)

        torch._dynamo.reset()
        # Exercise the upstream value-use conversion without changing scheduling.
        with (
            mock.patch.object(optimize_indexing, "indexing_dtype_strength_reduction", prepare),
            mock.patch.object(NPUTritonCSEVariable, "update_on_args", update),
        ):
            actual, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"}), x
            )
        self.assertEqual(actual, fn(x))
        self.assertIn("value_expr", seen)
        assertions = re.findall(r"tl\.static_assert\([^\n]*\.shape[^\n]*", "\n".join(codes))
        self.assertTrue(assertions)
        self.assertNotIn("XBLOCK", "\n".join(assertions))


    @dtypes(torch.float32)
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_unknown_load_shape_keeps_assertion(self, device, dtype):
        from torch._inductor.exc import InductorError
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonKernel

        x = torch.randn((5, 5, 5), device=device, dtype=dtype)
        mask = torch.ones((5, 5, 1), device=device, dtype=dtype)
        original_load = NPUTritonKernel.load

        def unknown_load(kernel, name, index):
            result = original_load(kernel, name, index)
            result._npu_load_index = None
            return result

        def fn(x, mask):
            return x * (mask * 2)

        torch._dynamo.reset()
        # Missing provenance must not remove the original compiler assertion.
        with mock.patch.object(NPUTritonKernel, "load", unknown_load):
            compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})
            with self.assertRaisesRegex(InductorError, "CompileTimeAssertionFailure"):
                compiled(x, mask)


instantiate_device_type_tests(TestLinearizeShape, globals(), only_for="npu")

if __name__ == "__main__":
    run_tests()
