import torch
import torch.nn.functional as F
from torch._inductor import config
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

    def test_dynamic_single_layer_norm_welford_fused(self):
        previous = npu_config.enable_welford
        npu_config.enable_welford = True
        torch._dynamo.reset()
        try:
            hidden = 512
            rows = self._generate_tensor((400, hidden), "float16")
            weight = self._generate_tensor((hidden,), "float16")
            bias = self._generate_tensor((hidden,), "float16")
            torch._dynamo.mark_dynamic(rows, 0, min=1, max=4094)

            def layer_norm(x, weight, bias):
                return F.layer_norm(x, (hidden,), weight, bias, 1e-6)

            expected = layer_norm(rows, weight, bias)
            compiled = torch.compile(layer_norm, backend="inductor", dynamic=True)
            actual, codes = run_and_get_code(compiled, rows, weight, bias)
            self.assertEqual(expected, actual, atol=1e-1, rtol=1e-1)
            code = "\n".join(codes)
            self.assertNotIn("triton_poi_fused_native_layer_norm", code)
        finally:
            npu_config.enable_welford = previous
            torch._dynamo.reset()

    def test_welford_full_static_mutation_hazard(self):
        previous = npu_config.enable_welford
        npu_config.enable_welford = True
        torch._dynamo.reset()
        try:
            def randn_fp16(shape, scale=0.1):
                return (
                    torch.randn(shape, device="npu", dtype=torch.float16) * scale
                )

            ids = torch.randint(
                0, 10000, (200, 1), device="npu", dtype=torch.int64
            )
            features = randn_fp16((200, 16))
            left_weight = randn_fp16((16, 32))
            left_bias = randn_fp16((32,))
            right_weight = randn_fp16((16, 32))
            right_bias = randn_fp16((32,))
            norm_weight = torch.ones((32,), device="npu", dtype=torch.float16)
            norm_weight = norm_weight + randn_fp16((32,))
            norm_bias = randn_fp16((32,))
            side_weight = randn_fp16((32, 16))
            side_bias = randn_fp16((16,))

            def layer_norm_with_reused_input(
                ids,
                features,
                left_weight,
                left_bias,
                right_weight,
                right_bias,
                norm_weight,
                norm_bias,
                side_weight,
                side_bias,
            ):
                left = torch.addmm(left_bias, features, left_weight)
                right = torch.addmm(right_bias, features, right_weight)
                condition = torch.logical_or(ids == 9998, ids == 3).repeat(1, 32)
                pre_norm = torch.where(condition, left, right)
                variance, mean = torch.var_mean(
                    pre_norm, dim=-1, correction=0, keepdim=True
                )
                normalized = (
                    (pre_norm - mean) * torch.rsqrt(variance + 1e-6) * norm_weight
                    + norm_bias
                )
                projected = torch.addmm(side_bias, pre_norm, side_weight)
                side_sum = pre_norm + normalized.sum(dim=-1, keepdim=True)
                return pre_norm, normalized, projected, side_sum

            args = (
                ids,
                features,
                left_weight,
                left_bias,
                right_weight,
                right_bias,
                norm_weight,
                norm_bias,
                side_weight,
                side_bias,
            )
            expected = layer_norm_with_reused_input(*args)
            compiled = torch.compile(
                layer_norm_with_reused_input,
                backend="inductor",
                dynamic=False,
                options={"unroll_reductions_threshold": 1},
            )
            with config.patch("triton.codegen_upcast_to_fp32", False):
                actual, codes = run_and_get_code(compiled, *args)

            for output in (*expected, *actual):
                self.assertTrue(torch.isfinite(output).all().item())

            for expected_output, actual_output in zip(expected, actual):
                self.assertEqual(
                    expected_output, actual_output, atol=1e-1, rtol=1e-1
                )

            actual_variance, actual_mean = torch.var_mean(
                actual[0], dim=-1, correction=0, keepdim=True
            )
            normalized_from_actual_input = (
                (actual[0] - actual_mean)
                * torch.rsqrt(actual_variance + 1e-6)
                * norm_weight
                + norm_bias
            )
            self.assertEqual(
                normalized_from_actual_input, actual[1], atol=1e-2, rtol=1e-2
            )

            code = "\n".join(codes)
            if npu_config.is_ascend950:
                self.assertIn("npu_kernel_type': 'simd'", code)
            self.assertIn("mutated_arg_names': ['in_out_ptr", code)
            self.assertNotIn("'vectorized_welford_axis':", code)
        finally:
            npu_config.enable_welford = previous
            torch._dynamo.reset()

    def test_welford_simd_codegen_above_persistent_threshold(self):
        if not npu_config.is_ascend950:
            self.skipTest("Welford SIMD rollout is Ascend 950-specific")
        previous = npu_config.enable_welford
        npu_config.enable_welford = True
        torch._dynamo.reset()
        try:
            input_element = self._generate_tensor((200, 5036), "float16")
            weight = self._generate_tensor((5036,), "float16")
            bias = self._generate_tensor((5036,), "float16")

            def layer_norm(x, gamma, beta):
                return F.layer_norm(x, (5036,), gamma, beta, 1e-6)

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
            self.assertIn("npu_kernel_type': 'simd'", code)
            self.assertIn("'vectorized_welford_axis':", code)
            self.assertNotIn("for loop_r", code)
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
