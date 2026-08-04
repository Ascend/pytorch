import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor.config as npu_config


class TestVarMean(TestUtils):
    def op_calc(self, input_element, dim):
        return torch.var_mean(input_element, dim)

    # case：The shape must not be too large
    @parametrize('shape', [(8, 64, 128)])
    @parametrize('dim', [0, 1, 2, (0, 2), (0, 1)])
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_shapes(self, shape, dim, dtype):

        input_element = self._generate_tensor(shape, dtype)

        std_var, std_mean = self.op_calc(input_element, dim)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=False)
        inductor_var, inductor_mean = compiled_op_calc(input_element, dim)

        self.assertEqual(std_var, inductor_var, atol=1e-1, rtol=1e-1, equal_nan=True)
        self.assertEqual(std_mean, inductor_mean, atol=1e-1, rtol=1e-1, equal_nan=True)

    def test_welford_enabled(self):
        previous = npu_config.enable_welford
        npu_config.enable_welford = True
        torch._dynamo.reset()
        try:
            input_element = self._generate_tensor((8, 64, 128), "float32")
            std_var, std_mean = self.op_calc(input_element, -1)

            compiled_op_calc = torch.compile(
                self.op_calc,
                backend="inductor",
                dynamic=False,
                options={"unroll_reductions_threshold": 1},
            )
            inductor_var, inductor_mean = compiled_op_calc(input_element, -1)

            self.assertEqual(std_var, inductor_var, atol=1e-1, rtol=1e-1, equal_nan=True)
            self.assertEqual(std_mean, inductor_mean, atol=1e-1, rtol=1e-1, equal_nan=True)
        finally:
            npu_config.enable_welford = previous
            torch._dynamo.reset()

    def test_welford_does_not_change_standalone_var_codegen(self):
        previous = npu_config.enable_welford
        npu_config.enable_welford = True
        torch._dynamo.reset()
        try:
            input_element = self._generate_tensor((8, 16, 128), "float32")

            def normalize(x):
                mean = torch.mean(x, dim=-1, keepdim=True)
                var = torch.var(x, dim=-1, correction=0, keepdim=True)
                return (x - mean) / torch.sqrt(var + 1e-6)

            expected = normalize(input_element)
            compiled = torch.compile(
                normalize,
                backend="inductor",
                dynamic=False,
                options={"unroll_reductions_threshold": 1},
            )
            actual, codes = run_and_get_code(compiled, input_element)

            self.assertEqual(expected, actual, atol=1e-1, rtol=1e-1)
            code = "\n".join(codes)
            self.assertNotIn("triton_helpers.welford_reduce", code)
            self.assertNotIn("triton_helpers.welford(", code)
        finally:
            npu_config.enable_welford = previous
            torch._dynamo.reset()

    def test_welford_simd_codegen(self):
        previous = npu_config.enable_welford
        npu_config.enable_welford = True
        torch._dynamo.reset()
        try:
            input_element = self._generate_tensor((381, 8, 80, 8), "float32")
            weight = self._generate_tensor((64,), "float32")
            bias = self._generate_tensor((64,), "float32")

            def layer_norm(x, gamma, beta):
                x = x.permute(0, 2, 1, 3).contiguous().view(381, 80, 64)
                return F.layer_norm(x, (64,), gamma, beta, 1e-6)

            expected = layer_norm(input_element, weight, bias)
            compiled = torch.compile(
                layer_norm,
                backend="inductor",
                dynamic=False,
                options={
                    "unroll_reductions_threshold": 1,
                    "aggressive_fusion": True,
                },
            )
            actual, codes = run_and_get_code(
                compiled, input_element, weight, bias
            )
            self.assertEqual(expected, actual, atol=1e-1, rtol=1e-1)

            code = "\n".join(codes)
            self.assertIn("for x1_loop_offset in range(", code)
            self.assertIn("X1BLOCK_SUB", code)
            self.assertIn("_acc_sum = tl.zeros([", code)
            self.assertIn(", 1], tl.float32)", code)
            self.assertIn("axis=1, keep_dims=True", code)
            self.assertIn("npu_kernel_type': 'simd'", code)
            self.assertIn("'vectorized_welford_axis':", code)
            self.assertIn("_mask = True", code)
            self.assertNotIn("for loop_r", code)
            self.assertNotIn("_acc_count = tl.zeros", code)
            self.assertNotIn("_acc_count += tl.where", code)
            self.assertEqual(code.count("tl.load(in_ptr0"), 1)
            rsqrt_markers = (" = tl.rsqrt(", " = libdevice.rsqrt(")
            rsqrt_positions = [
                code.index(marker) for marker in rsqrt_markers if marker in code
            ]
            self.assertTrue(rsqrt_positions)
            self.assertLess(
                code.index("axis=1, keep_dims=True"), min(rsqrt_positions)
            )
        finally:
            npu_config.enable_welford = previous
            torch._dynamo.reset()

    @parametrize("dtype", ["float16", "bfloat16"])
    def test_welford_simd_low_precision_codegen(self, dtype):
        if not npu_config.is_ascend950:
            self.skipTest("low-precision Welford fallback is Ascend 950-specific")

        previous = npu_config.enable_welford
        npu_config.enable_welford = True
        torch._dynamo.reset()
        try:
            input_element = self._generate_tensor((8, 16, 64), dtype)
            weight = self._generate_tensor((64,), dtype)
            bias = self._generate_tensor((64,), dtype)

            def layer_norm(x, gamma, beta):
                return F.layer_norm(x, (64,), gamma, beta, 1e-6)

            expected = layer_norm(input_element, weight, bias)
            compiled = torch.compile(
                layer_norm,
                backend="inductor",
                dynamic=False,
                options={"unroll_reductions_threshold": 1},
            )
            actual, codes = run_and_get_code(
                compiled, input_element, weight, bias
            )

            self.assertEqual(expected, actual, atol=1e-1, rtol=1e-1)
            code = "\n".join(codes)
            self.assertIn("'vectorized_welford_axis':", code)
            self.assertIn("npu_kernel_type': 'simd'", code)
        finally:
            npu_config.enable_welford = previous
            torch._dynamo.reset()

    def test_welford_simd_codegen_disabled(self):
        previous = npu_config.enable_welford
        npu_config.enable_welford = False
        torch._dynamo.reset()
        try:
            input_element = self._generate_tensor((8, 16, 64), "float32")
            weight = self._generate_tensor((64,), "float32")
            bias = self._generate_tensor((64,), "float32")

            def layer_norm(x, gamma, beta):
                return F.layer_norm(x, (64,), gamma, beta, 1e-6)

            expected = layer_norm(input_element, weight, bias)
            compiled = torch.compile(
                layer_norm,
                backend="inductor",
                dynamic=False,
                options={"unroll_reductions_threshold": 1},
            )
            actual, codes = run_and_get_code(
                compiled, input_element, weight, bias
            )
            self.assertEqual(expected, actual, atol=1e-1, rtol=1e-1)
            self.assertNotIn(
                "'vectorized_welford_axis':", "\n".join(codes)
            )
        finally:
            npu_config.enable_welford = previous
            torch._dynamo.reset()

    def test_welford_simd_tail_codegen(self):
        previous = npu_config.enable_welford
        npu_config.enable_welford = True
        torch._dynamo.reset()
        try:
            input_element = self._generate_tensor((8, 8, 81, 8), "float32")
            weight = self._generate_tensor((64,), "float32")
            bias = self._generate_tensor((64,), "float32")

            def layer_norm(x, gamma, beta):
                x = x.permute(0, 2, 1, 3).contiguous().view(8, 81, 64)
                return F.layer_norm(x, (64,), gamma, beta, 1e-6)

            expected = layer_norm(input_element, weight, bias)
            compiled = torch.compile(
                layer_norm,
                backend="inductor",
                dynamic=False,
                options={
                    "unroll_reductions_threshold": 1,
                    "aggressive_fusion": True,
                },
            )
            actual, codes = run_and_get_code(
                compiled, input_element, weight, bias
            )
            self.assertEqual(expected, actual, atol=1e-1, rtol=1e-1)

            code = "\n".join(codes)
            tail_masks = (
                "if 81 % X1BLOCK_SUB != 0:",
                "x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)",
            )
            self.assertTrue(any(mask in code for mask in tail_masks))
            self.assertNotIn("for loop_r", code)
            self.assertNotIn("_acc_count = tl.zeros", code)
            self.assertEqual(code.count("tl.load(in_ptr0"), 1)
        finally:
            npu_config.enable_welford = previous
            torch._dynamo.reset()


instantiate_parametrized_tests(TestVarMean)

if __name__ == "__main__":
    run_tests()
