import os
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._inductor.codegen.common import CSE
from torch._inductor.utils import IndentedBuffer, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, instantiate_parametrized_tests, parametrize, run_tests

import torch_npu
import torch_npu.testing  # noqa: F401


@instantiate_parametrized_tests
class TestShapeAssertFormatting(TestCase):
    @parametrize("shape", [(), ("R0_BLOCK",), (1, "R0_BLOCK"), ("XBLOCK", "R0_BLOCK")])
    def test_shape_assert_format(self, shape):
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonKernel

        buffer = IndentedBuffer()
        NPUTritonKernel.codegen_shape_assert(SimpleNamespace(_npu_linearize=False), buffer, "tmp0", shape)
        actual = buffer.getvalue()
        self.assertIn(f"len(tmp0.shape) == {len(shape)}", actual)
        for axis, dim in enumerate(shape):
            self.assertIn(f"tmp0.shape[{axis}] == {dim}", actual)
        self.assertNotIn("tmp0.shape ==", actual)

    @torch._inductor.config.patch({"test_configs.runtime_triton_shape_assert": True})
    def test_other_kernel_uses_original_check(self):
        from torch._inductor.codegen import common
        from torch_npu._inductor.triton_experimental.codegen import triton as codegen

        with (
            V.set_kernel_handler(SimpleNamespace()),
            mock.patch.object(codegen, "_original_check_shape") as original,
        ):
            buffer = IndentedBuffer()
            common.check_shape(buffer, "tmp0", (1,))
            original.assert_called_once_with(buffer, "tmp0", (1,))

    @torch._inductor.config.patch({"test_configs.runtime_triton_shape_assert": False})
    def test_disabled_shape_assert_uses_original_check(self):
        from torch._inductor.codegen import common
        from torch_npu._inductor.triton_experimental.codegen import triton as codegen

        kernel = object.__new__(codegen.NPUTritonKernel)
        with (
            V.set_kernel_handler(kernel),
            mock.patch.object(codegen, "_original_check_shape") as original,
        ):
            buffer = IndentedBuffer()
            common.check_shape(buffer, "tmp0", ())
            original.assert_called_once_with(buffer, "tmp0", ())

    def test_shape_patch_is_idempotent(self):
        from torch._inductor.codegen import common
        from torch_npu._inductor.triton_experimental.codegen import triton as codegen

        original = codegen._original_check_shape
        codegen._patch_npu_shape_assert()
        codegen._patch_npu_shape_assert()
        self.assertIs(common.check_shape, codegen._npu_check_shape)
        self.assertIs(codegen._original_check_shape, original)


class TestScalarConstantShape(TestCase):
    @dtypes(torch.float16, torch.float32)
    @parametrize("shape", [(), (1,)])
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_constant_shape(self, device, dtype, shape):
        torch._dynamo.reset()
        x = torch.empty(shape, device=device, dtype=dtype)
        fn = torch.compile(
            torch.zeros_like,
            fullgraph=True,
            options={"npu_backend": "triton_experimental"},
        )
        actual, codes = run_and_get_code(fn, x)
        self.assertEqual(actual, torch.zeros_like(x))
        source = "\n".join(codes)
        self.assertRegex(source, r"tl\.static_assert\(len\(tmp\d+\.shape\) == 0\)")
        self.assertIn("tl.static_assert", source)



@instantiate_parametrized_tests
class TestWideInt64ConstantMetadata(TestCase):
    @parametrize("ndim,numels,shape", [(0, (), ()), (1, (1,), ()), (2, (8, 4), (1, 1))])
    @parametrize("value", [-(2**40), 2**40])
    def test_constant_metadata(self, ndim, numels, shape, value):
        from torch_npu._inductor.triton_experimental.codegen.triton import (
            NPUTritonCSEVariable, NPUTritonKernelOverrides,
        )

        kernel = SimpleNamespace(
            current_node=None,
            cse=CSE(),
            compute=IndentedBuffer(),
            range_trees=[SimpleNamespace(numel=n, is_reduction=False, tensor_dim=i)
                         for i, n in enumerate(numels)],
            triton_tensor_ndim=lambda: ndim,
            create_cse_var=NPUTritonCSEVariable,
        )
        with V.set_kernel_handler(kernel):
            result = NPUTritonKernelOverrides.constant(value, torch.int64)
        self.assertIsInstance(result, NPUTritonCSEVariable)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, torch.int64)
        self.assertTrue(kernel._npu_has_wide_i64_const)
        self.assertIn(f"tl.full({list(shape)}, {value}, tl.int64)", kernel.compute.getvalue())


instantiate_device_type_tests(TestScalarConstantShape, globals(), only_for="npu")

if __name__ == "__main__":
    run_tests()
