import os

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, instantiate_parametrized_tests, parametrize, run_tests

import torch_npu  # noqa: F401
from torch_npu.utils._dynamo import register_inductor_npu


register_inductor_npu()


class TestNativeBatchNormSavedStats(TestCase):
    @parametrize("dtype,affine", [
        (dtype, affine)
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
        for affine in ("none", "weight", "bias", "both")
    ] + [(torch.float16, "mixed"), (torch.bfloat16, "mixed")])
    @torch._inductor.config.patch({
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
        "force_disable_caches": True,
    })
    def test_no_stats_dtypes(self, device, dtype, affine):
        self.assertTrue(torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::_native_batch_norm_legit.no_stats", "PrivateUse1"))
        values = [torch.sin(torch.arange(96, dtype=torch.float32)).reshape(4, 3, 8) * 2,
                  torch.tensor([0.75, 1.25, 0.5]), torch.tensor([-0.25, 0.5, 0.1])]
        values = [value.to(dtype) for value in values]
        if affine == "mixed":
            values[1:] = [value.float() for value in values[1:]]
        present = [True, affine in ("weight", "both", "mixed"), affine in ("bias", "both", "mixed")]
        eager = [value.to(device).requires_grad_() if use else None for value, use in zip(values, present)]
        native = [value.detach().clone().requires_grad_() if value is not None else None for value in eager]
        compiled = [value.detach().clone().requires_grad_() if value is not None else None for value in eager]
        cpu = [value.detach().cpu().float().requires_grad_() if value is not None else None for value in eager]
        eps = 0.01

        def fn(x, weight, bias):
            return torch.ops.aten._native_batch_norm_legit.no_stats(x, weight, bias, True, 0.3, eps)

        expected = fn(*eager)
        direct = torch.ops.aten.native_batch_norm.default(*native, None, None, True, 0.3, eps)
        reference = torch.nn.functional.batch_norm(cpu[0], None, None, cpu[1], cpu[2], True, 0.3, eps)
        variance = cpu[0].var(dim=(0, 2), correction=0)
        mean = cpu[0].mean(dim=(0, 2))
        atol, rtol = {torch.float16: (1e-5, 1e-3), torch.bfloat16: (1e-5, 1.6e-2),
                      torch.float32: (1.5e-5, 1.3e-5)}[dtype]
        with self.subTest(stage="native_alias"):
            self.assertEqual(expected, direct, atol=0, rtol=0)
        with self.subTest(stage="eager_cpu_output"):
            self.assertEqual(expected[0].cpu(), reference.to(dtype), atol=atol, rtol=rtol)
        with self.subTest(stage="eager_cpu_saved"):
            self.assertEqual(expected[1].cpu(), mean, atol=1.5e-5, rtol=1.3e-5)
            self.assertEqual(expected[2].cpu(), variance, atol=1.5e-5, rtol=1.3e-5)
        actual = torch.compile(fn, fullgraph=True)(*compiled)
        with self.subTest(stage="compiled_output"):
            self.assertEqual(actual[0], expected[0], atol=atol, rtol=rtol)
        with self.subTest(stage="compiled_saved"):
            self.assertEqual(actual[1:], expected[1:], atol=1.5e-5, rtol=1.3e-5)
        grad = torch.cos(torch.arange(96, dtype=torch.float32)).reshape(4, 3, 8).to(dtype)
        cpu_grads = torch.autograd.grad(reference, [arg for arg in cpu if arg is not None], grad.float())
        for stage, output, args in [("eager_grad", expected[0], eager), ("compiled_grad", actual[0], compiled)]:
            with self.subTest(stage=stage):
                grads = torch.autograd.grad(output, [arg for arg in args if arg is not None], grad.to(device))
                for value, reference_grad in zip(grads, cpu_grads):
                    grad_atol, grad_rtol = (1.5e-5, 1.3e-5) if value.dtype == torch.float32 else (atol, rtol)
                    self.assertEqual(value.cpu(), reference_grad.to(value.dtype), atol=grad_atol, rtol=grad_rtol)


    @torch._inductor.config.patch({
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
        "force_disable_caches": True,
    })
    def test_no_stats_training(self, device):
        x = (torch.arange(6, dtype=torch.float32).reshape(1, 2, 3) / 7).to(device).requires_grad_()
        compiled_x = x.detach().clone().requires_grad_()

        def fn(value):
            return torch.ops.aten._native_batch_norm_legit.no_stats(value, None, None, True, 0.5, 1e-5)

        expected = fn(x)
        actual = torch.compile(fn, fullgraph=True)(compiled_x)
        self.assertEqual(actual, expected, atol=1.5e-5, rtol=1.3e-5)
        grad = torch.linspace(-1, 1, 6, device=device).reshape(1, 2, 3)
        expected[0].backward(grad)
        actual[0].backward(grad)
        self.assertEqual(compiled_x.grad, x.grad, atol=1.5e-5, rtol=1.3e-5)

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("mixed", [False, True])
    @parametrize("training", [False, True])
    @torch._inductor.config.patch({
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
        "force_disable_caches": True,
    })
    def test_functional_statistics(self, device, dtype, mixed, training):
        stats_dtype = torch.float32 if mixed else dtype
        args = [
            (torch.sin(torch.arange(96, dtype=torch.float32)).reshape(4, 3, 8) * 2).to(device=device, dtype=dtype),
            torch.tensor([0.75, 1.25, 0.5], device=device, dtype=stats_dtype),
            torch.tensor([-0.25, 0.5, 0.1], device=device, dtype=stats_dtype),
            torch.tensor([-0.5, 0.75, 0.25], device=device, dtype=stats_dtype),
            torch.tensor([0.5, 1.5, 0.75], device=device, dtype=stats_dtype),
        ]
        original = [value.clone() for value in args[3:]]
        reference_args = [value.clone() for value in args]
        expected = torch.ops.aten._native_batch_norm_legit.default(
            *reference_args, training, 0.3, 0.01)
        expected = (*expected, *reference_args[3:])

        def fn(*values):
            return torch.ops.aten._native_batch_norm_legit_functional.default(
                *values, training, 0.3, 0.01)

        eager = fn(*args)
        actual = torch.compile(fn, fullgraph=True)(*args)
        atol, rtol = {torch.float16: (1e-3, 1e-3), torch.bfloat16: (1e-5, 1.6e-2),
                      torch.float32: (1.5e-5, 1.3e-5)}[dtype]
        for stage, result in [("eager", eager), ("compiled", actual)]:
            with self.subTest(stage=stage):
                self.assertEqual(len(result), 5)
                for value, reference in zip(result, expected):
                    self.assertEqual(value.dtype, reference.dtype)
                    self.assertEqual(value.shape, reference.shape)
                    self.assertEqual(value, reference, atol=atol, rtol=rtol)
        self.assertEqual(args[3:], original, atol=0, rtol=0)

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @torch._inductor.config.patch({
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
        "force_disable_caches": True,
    })
    def test_inference_out(self, device, dtype):
        x = torch.arange(24, device=device, dtype=dtype).reshape(3, 2, 4) / 7
        mean = torch.tensor([-0.5, 0.75], device=device, dtype=dtype)
        variance = torch.tensor([0.5, 1.5], device=device, dtype=dtype)

        def fn(value, out, saved_mean, saved_rstd):
            return torch.ops.aten.native_batch_norm.out(
                value, None, None, mean, variance, False, 0.3, 0.01,
                out=out, save_mean=saved_mean, save_invstd=saved_rstd)

        eager_buffers = [torch.empty_like(x), torch.empty_like(mean), torch.empty_like(variance)]
        compiled_buffers = [torch.empty_like(value) for value in eager_buffers]
        expected = fn(x, *eager_buffers)
        actual = torch.compile(fn, fullgraph=True)(x, *compiled_buffers)
        atol, rtol = {torch.float16: (1e-3, 1e-3), torch.bfloat16: (1e-5, 1.6e-2),
                      torch.float32: (1.5e-5, 1.3e-5)}[dtype]
        self.assertEqual(actual, expected, atol=atol, rtol=rtol)
        self.assertEqual(compiled_buffers, eager_buffers, atol=atol, rtol=rtol)
        for value, buffer in zip(actual, compiled_buffers):
            self.assertEqual(value.data_ptr(), buffer.data_ptr())
            self.assertEqual(value.dtype, buffer.dtype)

    @parametrize("training,dtype", [
        (False, torch.float32), (False, torch.float16), (False, torch.bfloat16),
        (True, torch.float32),
        (True, torch.float16), (True, torch.bfloat16),
    ])
    @parametrize("op_name", ["native_batch_norm", "_native_batch_norm_legit"])
    @torch._inductor.config.patch({
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
        "force_disable_caches": True,
    })
    def test_saved_statistics(self, device, training, dtype, op_name):
        values = [
            torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) / 7,
            torch.tensor([0.75, 1.25]),
            torch.tensor([-0.25, 0.5]),
            torch.tensor([-0.5, 0.75]),
            torch.tensor([0.5, 1.5]),
        ]
        eager_args = [value.to(device=device, dtype=dtype) for value in values]
        compiled_args = [value.clone() for value in eager_args]
        for args in (eager_args, compiled_args):
            for value in args[:3]:
                value.requires_grad_(True)
        op = getattr(torch.ops.aten, op_name).default

        def fn(*args):
            return op(*args, training, 0.3, 0.01)

        expected = fn(*eager_args)
        actual = torch.compile(fn, fullgraph=True)(*compiled_args)
        atol, rtol = {torch.float16: (1e-5, 1e-3), torch.bfloat16: (1e-5, 1.6e-2),
                      torch.float32: (1.5e-5, 1.3e-5)}[dtype]
        self.assertEqual(actual, expected, atol=atol, rtol=rtol)
        self.assertEqual(compiled_args[3:], eager_args[3:], atol=atol, rtol=rtol)
        grad = torch.linspace(-1, 1, 24, device=device, dtype=dtype).reshape(3, 2, 4)
        expected[0].backward(grad)
        actual[0].backward(grad)
        for eager, compiled in zip(eager_args[:3], compiled_args[:3]):
            self.assertEqual(compiled.grad, eager.grad, atol=atol, rtol=rtol)


@instantiate_parametrized_tests
class TestNativeBatchNormCPUGate(TestCase):
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_no_stats_decomposition(self, dtype):
        from torch._decomp.decompositions import native_batch_norm_helper
        from torch_npu._inductor.decomposition import decompositions

        x = torch.arange(24, dtype=dtype).reshape(3, 2, 4) / 7
        expected = native_batch_norm_helper(x, None, None, None, None, True, 0.3, 0.01, False)[:3]
        actual = decompositions[torch.ops.aten._native_batch_norm_legit.no_stats](
            x, None, None, True, 0.3, 0.01)
        self.assertEqual(actual, expected, atol=0, rtol=0)


instantiate_device_type_tests(TestNativeBatchNormSavedStats, globals(), only_for=("npu",))

if __name__ == "__main__":
    run_tests()
