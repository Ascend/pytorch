# Owner(s): ["module: tests"]
import os

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, a, b, c):
        b = torch.transpose(b, 0, 1)
        add = a + b
        sub = c - a
        mul = add * sub
        mul = mul + 3
        return mul, torch.sum(mul, dim=[0, 2], keepdim=True)


class ReduceCaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, arg0_1, arg1_1, arg2_1):
        sum_1 = torch.ops.aten.sum.dim_IntList(arg0_1, [0, 2, 3], True)
        div = torch.ops.aten.div.Scalar(sum_1, 9800.0)
        view = torch.ops.aten.reshape.default(div, [64])
        mul = torch.ops.aten.mul.Scalar(view, 0.1)
        mul_1 = torch.ops.aten.mul.Scalar(arg1_1, 0.9)
        add = torch.ops.aten.add.Tensor(mul, mul_1)
        expand = torch.ops.aten.expand.default(div, [8, 64, 35, 35])
        sub = torch.ops.aten.sub.Tensor(arg0_1, expand)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_1, [0, 2, 3], True)
        div_1 = torch.ops.aten.div.Scalar(sum_2, 9800.0)
        add_1 = torch.ops.aten.add.Scalar(div_1, 0.001)
        rsqrt = torch.ops.aten.rsqrt.default(add_1)
        view_1 = torch.ops.aten.reshape.default(div_1, [64])
        mul_2 = torch.ops.aten.mul.Scalar(view_1, 1.0001020512297174)
        mul_3 = torch.ops.aten.mul.Scalar(mul_2, 0.1)
        mul_4 = torch.ops.aten.mul.Scalar(arg2_1, 0.9)
        add_2 = torch.ops.aten.add.Tensor(mul_3, mul_4)
        return (div, add, rsqrt, add_2)


class DeterministicReduceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, arg0):
        return torch.ops.aten.sum.default(arg0)


class BitwiseBoolModel(torch.nn.Module):
    def forward(self, arg0, arg1):
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(arg0, arg1)
        bitwise_not = torch.ops.aten.bitwise_not.default(arg0)
        return torch.ops.aten.bitwise_or.Tensor(bitwise_and, bitwise_not)


class BitwiseIntModel(torch.nn.Module):
    def forward(self, arg0):
        return torch.ops.aten.bitwise_not.default(arg0)


class MmTransposeBackwardModel(torch.nn.Module):
    def forward(self, a, b):
        loss = torch.mm(a, b.t()).sum()
        grad_a, grad_b = torch.autograd.grad(loss, (a, b))
        return loss, grad_a, grad_b


class MmTemplateModel(torch.nn.Module):
    def forward(self, a, b, residual):
        return (torch.mm(a, b) + residual) * 0.5


class BmmTemplateModel(torch.nn.Module):
    def forward(self, a, b, residual):
        return torch.bmm(a, b) + residual


class AddmmTemplateModel(torch.nn.Module):
    def forward(self, bias, a, b, residual):
        addmm = torch.addmm(bias, a.permute(1, 0), b.permute(1, 0))
        return addmm + residual


class BaddbmmTemplateModel(torch.nn.Module):
    def forward(self, bias, a, b, residual):
        return torch.baddbmm(bias, a, b) + residual


class CopyInplaceModel(torch.nn.Module):
    def forward(self, dst, src):
        add = torch.ops.aten.add.Tensor(src, 1.0)
        torch.ops.aten.copy_.default(dst, add)
        return ()


class CopyModel(torch.nn.Module):
    def forward(self, dst, src):
        copied = torch.ops.aten.copy.default(dst, src)
        return torch.ops.aten.add.Tensor(copied, 1.0)


class Int64AddModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class Int64CompareModel(torch.nn.Module):
    def forward(self, x, y):
        return x > y


class Int64PointwiseFusionModel(torch.nn.Module):
    def forward(self, x, y, ids):
        sum_xy = x + y
        mask = sum_xy > y
        pointwise = torch.where(mask, sum_xy, x)
        dense_mask = torch.unsqueeze(
            torch.where(
                ids >= 0,
                torch.ones_like(ids, dtype=torch.float32),
                torch.zeros_like(ids),
            ),
            dim=-1,
        )
        ids = torch.where(ids == -1, torch.zeros_like(ids), ids)
        return pointwise, dense_mask, ids


class TestDvmByMlir(TestCase):
    def _run_and_get_code_with_dvm(
        self, model, *args, dynamic=False, options=None, run_count=1
    ):
        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "dvm"
        os.environ["INDUCTOR_DVM_ENABLE_MATMUL_FUSION"] = "1"
        compiled_model = torch.compile(
            model, backend="inductor", dynamic=dynamic, options=options
        )
        result = run_and_get_code(compiled_model, *args)
        if run_count > 1:
            outputs, codes = result
            outputs = [outputs]
            for _ in range(run_count - 1):
                outputs.append(compiled_model(*args))
            result = outputs, codes
        os.environ.pop("TORCHINDUCTOR_NPU_BACKEND")
        os.environ.pop("INDUCTOR_DVM_ENABLE_MATMUL_FUSION")
        return result

    def test_int64_add_fuses_into_dvm(self):
        arg0 = torch.randint(-8, 8, (32, 32), dtype=torch.int64, device="npu")
        arg1 = torch.randint(-8, 8, (32, 32), dtype=torch.int64, device="npu")
        model = Int64AddModel()

        with torch.no_grad():
            expect = model(arg0, arg1)
            result, codes = self._run_and_get_code_with_dvm(model, arg0, arg1)

        code = "\n".join(codes)
        self.assertEqual(expect, result)
        self.assertIn("k.add", code)

    def test_silu_bf16_matches_aclnn_bitwise(self):
        def model(grad, x):
            return (
                torch.ops.aten.silu.default(x),
                torch.ops.aten.silu_backward.default(grad, x),
            )

        numel = 257 * 263
        x = torch.linspace(-20.0, 20.0, numel, dtype=torch.float32)
        grad = torch.linspace(-2.0, 2.0, numel, dtype=torch.float32).cos()
        x = x.to(torch.bfloat16).reshape(257, 263).npu()
        grad = grad.to(torch.bfloat16).reshape(257, 263).npu()

        with torch.no_grad():
            expect = model(grad, x)
            result, codes = self._run_and_get_code_with_dvm(model, grad, x)

        self.assertTrue(torch.equal(expect[0], result[0]))
        self.assertTrue(torch.equal(expect[1], result[1]))
        self.assertIn("@dvm.kernel", "\n".join(codes))

    def test_int64_compare_fuses_into_dvm(self):
        arg0 = torch.randint(-8, 8, (32, 32), dtype=torch.int64, device="npu")
        arg1 = torch.randint(-8, 8, (32, 32), dtype=torch.int64, device="npu")
        model = Int64CompareModel()

        with torch.no_grad():
            expect = model(arg0, arg1)
            result, codes = self._run_and_get_code_with_dvm(model, arg0, arg1)

        code = "\n".join(codes)
        self.assertEqual(expect, result)
        self.assertIn("k.greater", code)

    def test_int64_pointwise_chain_fuses_into_dvm(self):
        arg0 = torch.randint(-8, 8, (32, 32), dtype=torch.int64, device="npu")
        arg1 = torch.randint(-8, 8, (32, 32), dtype=torch.int64, device="npu")
        ids = torch.randint(-2, 4, (32, 32), dtype=torch.int64, device="npu")
        model = Int64PointwiseFusionModel()

        with torch.no_grad():
            expect = model(arg0, arg1, ids)
            result, codes = self._run_and_get_code_with_dvm(model, arg0, arg1, ids)

        code = "\n".join(codes)
        self.assertEqual(expect, result)
        self.assertIn("k.add", code)
        self.assertIn("k.greater", code)
        self.assertIn("k.greater_equal", code)
        self.assertIn("k.equal", code)
        self.assertIn("k.select(", code)


    @parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    @parametrize("is_dynamic", [True, False])
    def test_basic_partitioning(self, dtype, is_dynamic):
        a = torch.normal(0, 0.01, size=(512, 1), dtype=dtype).npu()
        b = torch.normal(0, 0.01, size=(512, 4, 256), dtype=dtype).npu()
        c = torch.normal(0, 0.01, size=(1, 256), dtype=dtype).npu()
        model = TestModule()
        with torch.no_grad():
            expect = model(a, b, c)
            result, _ = self._run_and_get_code_with_dvm(
                model, a, b, c, dynamic=is_dynamic
            )
            self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)

    @parametrize("dtype", [torch.bfloat16])
    @parametrize("is_dynamic", [False])
    def test_basic_partitioning_npugraph(self, dtype, is_dynamic):
        a = torch.normal(0, 0.01, size=(512, 1), dtype=dtype).npu()
        b = torch.normal(0, 0.01, size=(512, 4, 256), dtype=dtype).npu()
        c = torch.normal(0, 0.01, size=(1, 256), dtype=dtype).npu()
        model = TestModule()
        with torch.no_grad():
            expect = model(a, b, c)
            results, _ = self._run_and_get_code_with_dvm(
                model,
                a,
                b,
                c,
                dynamic=is_dynamic,
                options={"triton.cudagraphs": True},
                run_count=3,
            )
            self.assertEqual(expect, results[-1], atol=1e-3, rtol=1e-3)

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("is_dynamic", [True, False])
    def test_reduce_case(self, dtype, is_dynamic):
        arg0 = torch.empty_strided(
            torch.Size((8, 64, 35, 35)),
            (78400, 1225, 35, 1),
            dtype=dtype,
            device="npu",
        ).uniform_(0, 1)
        arg1 = torch.empty_strided(
            torch.Size((64,)), (1,), dtype=dtype, device="npu"
        ).uniform_(0, 1)
        arg2 = torch.empty_strided(
            torch.Size((64,)), (1,), dtype=dtype, device="npu"
        ).uniform_(0, 1)
        model = ReduceCaseModel()
        with torch.no_grad():
            expect = model(arg0, arg1, arg2)
            result, _ = self._run_and_get_code_with_dvm(
                model, arg0, arg1, arg2, dynamic=is_dynamic
            )
            self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)

    def test_deterministic_reduce_case(self):
        deterministic_state = torch.are_deterministic_algorithms_enabled()
        deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        arg0 = torch.normal(
            0, 0.1, size=(16, 128, 64, 64), dtype=torch.float32, device="npu"
        )
        model = DeterministicReduceModel()
        try:
            torch.use_deterministic_algorithms(True)
            with torch.no_grad():
                results, _ = self._run_and_get_code_with_dvm(
                    model, arg0, run_count=2
                )
                first_result, second_result = results
                self.assertEqual(first_result, second_result, atol=0, rtol=0)
        finally:
            torch.use_deterministic_algorithms(
                deterministic_state, warn_only=deterministic_warn_only
            )

    def test_bitwise_bool_ops_codegen(self):
        arg0 = torch.randint(0, 2, (32, 32), dtype=torch.bool, device="npu")
        arg1 = torch.randint(0, 2, (32, 32), dtype=torch.bool, device="npu")
        model = BitwiseBoolModel()

        with torch.no_grad():
            expect = model(arg0, arg1)
            result, codes = self._run_and_get_code_with_dvm(model, arg0, arg1)

        code = "\n".join(codes)
        self.assertEqual(expect, result)
        self.assertIn("k.logical_and", code)
        self.assertIn("k.logical_or", code)
        self.assertIn("k.logical_not", code)

    def test_bitwise_int_rule_fallback(self):
        arg0 = torch.randint(-8, 8, (32, 32), dtype=torch.int32, device="npu")
        model = BitwiseIntModel()

        with torch.no_grad():
            expect = model(arg0)
            result, codes = self._run_and_get_code_with_dvm(model, arg0)

        code = "\n".join(codes)
        self.assertEqual(expect, result)
        self.assertNotIn("k.logical_not", code)

    def test_mm_t_backward_no_dvm_fused_matmul_backward(self):
        a = torch.randn((4, 8), dtype=torch.float32, device="npu")
        b = torch.randn((3, 8), dtype=torch.float32, device="npu")
        a_eager = a.detach().clone().requires_grad_(True)
        b_eager = b.detach().clone().requires_grad_(True)
        a_compiled = a.detach().clone().requires_grad_(True)
        b_compiled = b.detach().clone().requires_grad_(True)
        model = MmTransposeBackwardModel()

        expect = model(a_eager, b_eager)
        result, codes = self._run_and_get_code_with_dvm(
            model, a_compiled, b_compiled
        )

        code = "\n".join(codes)
        self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)
        self.assertNotIn("dvm_fused_matmul_backward", code)

    @parametrize("op", ["mm", "bmm", "addmm", "baddbmm"])
    def test_matmul_uses_dvm_fusion(self, op):
        if op == "addmm":
            a_shape = (128, 256)
            b_shape = (1024, 128)
            output_shape = (256, 1024)
        elif op in ("bmm", "baddbmm"):
            a_shape = (2, 128, 64)
            b_shape = (2, 64, 128)
            output_shape = (2, 128, 128)
        else:
            a_shape = (64, 128)
            b_shape = (128, 512)
            output_shape = (64, 512)
        a = torch.normal(
            0, 0.1, size=a_shape, dtype=torch.float16, device="npu"
        )
        b = torch.normal(
            0, 0.1, size=b_shape, dtype=torch.float16, device="npu"
        )
        residual = torch.normal(
            0, 0.1, size=output_shape, dtype=torch.float16, device="npu"
        )
        if op == "mm":
            model = MmTemplateModel()
            model_args = (a, b, residual)
        elif op == "bmm":
            model = BmmTemplateModel()
            model_args = (a, b, residual)
        elif op == "addmm":
            bias = torch.normal(
                0,
                0.1,
                size=(output_shape[-1],),
                dtype=torch.float16,
                device="npu",
            )
            model = AddmmTemplateModel()
            model_args = (bias, a, b, residual)
        else:
            bias = torch.normal(
                0, 0.1, size=output_shape, dtype=torch.float16, device="npu"
            )
            model = BaddbmmTemplateModel()
            model_args = (bias, a, b, residual)
        with torch.no_grad():
            expect = model(*model_args)
            result, codes = self._run_and_get_code_with_dvm(model, *model_args)

        code = "\n".join(codes)
        self.assertEqual(expect, result, atol=5e-3, rtol=5e-3)
        self.assertIn("k.matmul(", code)

    @parametrize("op", ["mm", "bmm"])
    def test_k1_matmul_lowers_to_mul(self, op):
        if op == "mm":
            a_shape = (64, 1)
            b_shape = (1, 512)

            def model(lhs, rhs):
                return torch.mm(lhs, rhs)
        elif op == "bmm":
            a_shape = (2, 64, 1)
            b_shape = (2, 1, 512)

            def model(lhs, rhs):
                return torch.bmm(lhs, rhs)

        a = torch.normal(0, 0.1, size=a_shape, dtype=torch.float16, device="npu")
        b = torch.normal(0, 0.1, size=b_shape, dtype=torch.float16, device="npu")

        with torch.no_grad():
            expect = model(a, b)
            result, codes = self._run_and_get_code_with_dvm(model, a, b)

        code = "\n".join(codes)
        self.assertEqual(expect, result, atol=5e-3, rtol=5e-3)
        self.assertIn("k.mul(", code)
        self.assertNotIn("k.matmul(", code)

    def test_k1_addmm_lowers_to_pointwise(self):
        a = torch.normal(
            0, 0.1, size=(64, 1), dtype=torch.float16, device="npu"
        )
        b = torch.normal(
            0, 0.1, size=(1, 512), dtype=torch.float16, device="npu"
        )
        bias = torch.normal(
            0, 0.1, size=(512,), dtype=torch.float16, device="npu"
        )

        def model(lhs, rhs):
            return torch.addmm(bias, lhs, rhs)

        with torch.no_grad():
            expect = model(a, b)
            result, codes = self._run_and_get_code_with_dvm(model, a, b)

        code = "\n".join(codes)
        self.assertEqual(expect, result, atol=5e-3, rtol=5e-3)
        self.assertIn("k.mul(", code)
        self.assertNotIn("k.matmul(", code)

    def test_matmul_fusion_output_with_multiple_users(self):
        def model(a, b, denom, scale):
            mm = torch.mm(a, b).reshape(4, 8, 16)
            reduced = mm.float().sum(dim=(0, 1), keepdim=True)
            scaled_reduced = ((mm / denom) * scale).float().sum(
                dim=(0, 1), keepdim=True
            )
            return reduced, scaled_reduced

        a = torch.normal(
            0, 0.01, size=(32, 64), dtype=torch.float16, device="npu"
        )
        b = torch.normal(
            0, 0.01, size=(64, 16), dtype=torch.float16, device="npu"
        )
        denom = torch.rand((4, 8, 1), dtype=torch.float16, device="npu") + 0.5
        scale = torch.normal(
            0, 0.01, size=(4, 8, 16), dtype=torch.float16, device="npu"
        )
        with torch.no_grad():
            expect = model(a, b, denom, scale)
            result, codes = self._run_and_get_code_with_dvm(
                model, a, b, denom, scale
            )

        code = "\n".join(codes)
        self.assertEqual(expect, result, atol=1e-2, rtol=1e-2)
        self.assertIn("k.matmul(", code)

    def test_matmul_does_not_fuse_view_only_epilogue(self):
        def model(a, b):
            return torch.mm(a, b).reshape(8, 8, 512)

        a = torch.normal(
            0, 0.01, size=(64, 128), dtype=torch.float16, device="npu"
        )
        b = torch.normal(
            0, 0.01, size=(128, 512), dtype=torch.float16, device="npu"
        )
        with torch.no_grad():
            expect = model(a, b)
            result, codes = self._run_and_get_code_with_dvm(model, a, b)

        code = "\n".join(codes)
        self.assertEqual(expect, result, atol=5e-3, rtol=5e-3)
        self.assertIn("k.matmul(", code)

    def test_matmul_fuses_view_with_pointwise_epilogue(self):
        def model(a, b, residual):
            view = torch.mm(a, b).reshape(8, 8, 512)
            return view + residual

        a = torch.normal(
            0, 0.01, size=(64, 128), dtype=torch.float16, device="npu"
        )
        b = torch.normal(
            0, 0.01, size=(128, 512), dtype=torch.float16, device="npu"
        )
        residual = torch.normal(
            0, 0.01, size=(8, 8, 512), dtype=torch.float16, device="npu"
        )
        with torch.no_grad():
            expect = model(a, b, residual)
            result, codes = self._run_and_get_code_with_dvm(model, a, b, residual)

        code = "\n".join(codes)
        self.assertEqual(expect, result, atol=5e-3, rtol=5e-3)
        self.assertIn("k.matmul(", code)

    def test_bmm_with_view_input_uses_dvm_fusion(self):
        def model(a, b):
            softmax = torch.softmax(a, dim=-1)
            return torch.bmm(softmax.reshape(4, 128, 128), b)

        a = torch.normal(
            0, 0.01, size=(2, 2, 128, 128), dtype=torch.float16, device="npu"
        )
        b = torch.normal(
            0, 0.01, size=(4, 128, 64), dtype=torch.float16, device="npu"
        )
        with torch.no_grad():
            expect = model(a, b)
            result, codes = self._run_and_get_code_with_dvm(model, a, b)

        code = "\n".join(codes)
        self.assertEqual(expect, result, atol=5e-3, rtol=5e-3)
        self.assertIn("k.matmul(", code)

    def test_bmm_same_buffer_views_keep_distinct_input_meta(self):
        def model(x):
            lhs = x.reshape(4, 64, 128)
            rhs = x.reshape(4, 128, 64)
            return torch.bmm(lhs, rhs)

        x = torch.normal(
            0, 0.01, size=(4, 8192), dtype=torch.float16, device="npu"
        )
        with torch.no_grad():
            expect = model(x)
            result, codes = self._run_and_get_code_with_dvm(model, x)

        code = "\n".join(codes)
        self.assertEqual(expect, result, atol=5e-3, rtol=5e-3)
        self.assertIn("k.matmul(", code)

    def test_copy_inplace_codegen(self):
        src = torch.randn((128,), dtype=torch.float32, device="npu")
        dst = torch.zeros((128,), dtype=torch.float32, device="npu")
        expect_dst = dst.clone()
        actual_dst = dst.clone()
        model = CopyInplaceModel()

        with torch.no_grad():
            model(expect_dst, src)
            _, codes = self._run_and_get_code_with_dvm(model, actual_dst, src)

        self.assertEqual(expect_dst, actual_dst)
        self.assertIn("@dvm.kernel", "\n".join(codes))

    def test_copy_codegen(self):
        src = torch.randn((128,), dtype=torch.float32, device="npu")
        dst = torch.zeros((128,), dtype=torch.float32, device="npu")
        model = CopyModel()

        with torch.no_grad():
            expect = model(dst, src)
            result, codes = self._run_and_get_code_with_dvm(model, dst, src)

        self.assertEqual(expect, result)
        self.assertIn("@dvm.kernel", "\n".join(codes))


instantiate_parametrized_tests(TestDvmByMlir)
if __name__ == "__main__":
    run_tests()
