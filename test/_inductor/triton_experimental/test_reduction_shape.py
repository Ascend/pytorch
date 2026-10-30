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


instantiate_device_type_tests(TestReductionShape, globals(), only_for="npu")

if __name__ == "__main__":
    run_tests()
