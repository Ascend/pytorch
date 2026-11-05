import re
from unittest import mock

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests

import torch_npu
import torch_npu.testing  # noqa: F401


class TestPermutedReductionResultShape(TestCase):
    @dtypes(torch.float16, torch.float32)
    @parametrize("transpose", [False, True])
    @torch._inductor.config.patch({
        "force_disable_caches": True,
        "triton.persistent_reductions": False,
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
    })
    def test_reduction_cast(self, device, dtype, transpose):
        from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonKernel

        x = torch.randn((10, 5) if transpose else (5, 10), device=device, dtype=dtype)
        if transpose:
            x = x.t()
        x.requires_grad_()
        weight = torch.randn(10, device=device, dtype=dtype, requires_grad=True)
        original = NPUTritonKernel.codegen_kernel
        original_reduction = NPUTritonKernel.reduction
        reduction_results = []
        permuted_results = []
        permuted_casts = []

        def reduction(kernel, *args, **kwargs):
            result = original_reduction(kernel, *args, **kwargs)
            axes = [tree.tensor_dim for tree in kernel.range_trees if tree.is_reduction]
            if getattr(kernel, "_npu_tile_permuted", False) and axes == [0]:
                values = result if isinstance(result, tuple) else (result,)
                reduction_results.extend((kernel, value) for value in values)
            return result

        def codegen(kernel, *args, **kwargs):
            source = original(kernel, *args, **kwargs)
            for owner, result in reduction_results:
                if owner is not kernel:
                    continue
                shape = kernel._npu_assert_shapes.get(result.name)
                self.assertIsNotNone(shape)
                self.assertEqual(tuple(map(str, shape)), ("1", "real_block_x0"))
                permuted_results.append(result.name)
                for var in kernel.cse.varname_map.values():
                    op = getattr(var, "_npu_shape_op", None)
                    if op is None or op[0] != "to_dtype" or not op[1] or op[1][0] is not result:
                        continue
                    cast_shape = kernel._npu_assert_shapes.get(var.name)
                    self.assertIsNotNone(cast_shape)
                    self.assertEqual(tuple(map(str, cast_shape)), ("1", "real_block_x0"))
                    self.assertIn(f"{var.name} = {result.name}.to(", source)
                    name = re.escape(var.name)
                    self.assertRegex(
                        source,
                        rf"tl\.static_assert\([^\n]*len\({name}\.shape\) == 2"
                        rf"[^\n]*{name}\.shape\[0\] == 1[^\n]*{name}\.shape\[1\] == real_block_x0",
                    )
                    permuted_casts.append(var.name)
            return source

        def fn(x, weight):
            return x @ weight

        torch._dynamo.reset()
        with (
            mock.patch.object(NPUTritonKernel, "reduction", reduction),
            mock.patch.object(NPUTritonKernel, "codegen_kernel", codegen),
        ):
            actual, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"}), x, weight
            )
            actual_grads = torch.autograd.grad(actual.sum(), (x, weight))
        expected = fn(x, weight)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_grads, torch.autograd.grad(expected.sum(), (x, weight)))
        self.assertIn("tl.static_assert", "\n".join(codes))
        # Strides select permutation; fp16 additionally exercises the result cast.
        # The other combinations remain numerical controls, not non-permutation claims.
        if transpose and dtype == torch.float16:
            self.assertTrue(permuted_results, "expected an axis-zero permuted reduction")
            self.assertTrue(permuted_casts, "expected a cast consuming the permuted result")


instantiate_device_type_tests(TestPermutedReductionResultShape, globals(), only_for="npu")

if __name__ == "__main__":
    run_tests()
