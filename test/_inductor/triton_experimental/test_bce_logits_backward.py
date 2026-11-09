import os

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, instantiate_parametrized_tests, parametrize, run_tests

import torch_npu  # noqa: F401
import torch_npu.testing  # noqa: F401
from torch_npu.utils._dynamo import register_inductor_npu


register_inductor_npu()


class TestBCELogitsBackward(TestCase):
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("reduction", ["none", "mean", "sum"])
    @parametrize("weighted", [False, True])
    @torch._inductor.config.patch({
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
        "force_disable_caches": True,
        "implicit_fallbacks": False,
    })
    def test_forward_backward(self, device, dtype, reduction, weighted):
        values = torch.linspace(-3, 3, 12).reshape(3, 4).to(dtype)
        targets = torch.linspace(0.1, 0.9, 12).reshape(3, 4).to(dtype)
        weight = torch.tensor([[0.75], [1.25], [0.5]], dtype=dtype) if weighted else None
        pos_weight = torch.tensor([0.5, 1.5, 0.75, 2], dtype=dtype) if weighted else None

        def loss(x, target, weight, pos_weight):
            return torch.nn.functional.binary_cross_entropy_with_logits(
                x, target, weight=weight, pos_weight=pos_weight, reduction=reduction
            )

        eager = [values.to(device).requires_grad_(), targets.to(device)]
        compiled = [eager[0].detach().clone().requires_grad_(), eager[1].detach().clone()]
        reference = [values.float().requires_grad_(), targets.float()]
        device_weights = [value.to(device) if value is not None else None for value in (weight, pos_weight)]
        cpu_weights = [value.float() if value is not None else None for value in (weight, pos_weight)]
        expected = loss(*eager, *device_weights)
        cpu_output = loss(*reference, *cpu_weights)
        actual = torch.compile(loss, fullgraph=True)(*compiled, *device_weights)
        atol, rtol = {torch.float16: (1e-3, 1e-3), torch.bfloat16: (1e-2, 1.6e-2),
                      torch.float32: (1e-5, 1.3e-5)}[dtype]
        self.assertEqual(actual, expected, atol=atol, rtol=rtol)
        self.assertEqual(actual.cpu(), cpu_output.to(dtype), atol=atol, rtol=rtol)
        grad = torch.linspace(-0.75, 1.25, actual.numel()).reshape(actual.shape).to(dtype)
        cpu_grads = torch.autograd.grad(cpu_output, reference[:1], grad.float())
        eager_grads = torch.autograd.grad(expected, eager[:1], grad.to(device))
        compiled_grads = torch.autograd.grad(actual, compiled[:1], grad.to(device))
        for actual_grad, eager_grad, cpu_grad in zip(compiled_grads, eager_grads, cpu_grads):
            self.assertEqual(actual_grad, eager_grad, atol=atol, rtol=rtol)
            self.assertEqual(actual_grad.cpu(), cpu_grad.to(dtype), atol=atol, rtol=rtol)


    # The native forward is gone: these assert on the generated code so a silent
    # relapse to a fallback (native BCE, or an extern log1p splitting the kernel)
    # fails here instead of quietly costing the fusion.
    NATIVE_CALLS = (
        "torch.ops.aten.binary_cross_entropy_with_logits",
        "torch.ops.npu.npu_binary_cross_entropy_with_logits",
        "torch.ops.aten.log1p",
        "torch.ops.aten.log_sigmoid",
    )

    def _assert_no_native_call(self, code):
        # The output code repeats the whole FX graph in a comment block, where
        # every node is spelled torch.ops.aten.*; only real statements count.
        for generated in code:
            statements = "\n".join(
                line for line in generated.splitlines() if not line.lstrip().startswith("#")
            )
            for call in self.NATIVE_CALLS:
                self.assertNotIn(call, statements)

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("reduction", ["none", "mean"])
    @parametrize("weighted", [False, True])
    @torch._inductor.config.patch({
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
        "force_disable_caches": True,
        "implicit_fallbacks": False,
    })
    def test_generated_code_is_fused(self, device, dtype, reduction, weighted):
        from torch._inductor.utils import run_and_get_code

        values = torch.linspace(-3, 3, 12).reshape(3, 4).to(dtype).to(device).requires_grad_()
        targets = torch.linspace(0.1, 0.9, 12).reshape(3, 4).to(dtype).to(device)
        weight = torch.tensor([[0.75], [1.25], [0.5]], dtype=dtype, device=device) if weighted else None
        pos_weight = torch.tensor([0.5, 1.5, 0.75, 2], dtype=dtype, device=device) if weighted else None

        def loss(x):
            return torch.nn.functional.binary_cross_entropy_with_logits(
                x, targets, weight=weight, pos_weight=pos_weight, reduction=reduction
            )

        compiled = torch.compile(loss, fullgraph=True)
        actual, forward_code = run_and_get_code(compiled, values)
        self._assert_no_native_call(forward_code)
        # The loss math itself must live in a single kernel. Counting every
        # kernel in the graph would instead measure inductor's input handling:
        # with a weight/pos_weight it materializes the dtype-converted broadcast
        # operand into its own kernel, which carries no BCE math at all (the same
        # split appears in the untouched backward graph).
        math_kernels = sum(
            1
            for generated in forward_code
            for kernel in generated.split("async_compile.triton(")[1:]
            if "libdevice.log1p" in kernel
        )
        self.assertEqual(math_kernels, 1)
        if reduction == "none" and not weighted:
            # Nothing to materialize here, so the whole forward is one kernel.
            self.assertEqual(sum(c.count("async_compile.triton(") for c in forward_code), 1)
        grad = torch.ones_like(actual)
        _, backward_code = run_and_get_code(
            lambda: torch.autograd.grad(actual, values, grad, retain_graph=True)
        )
        self._assert_no_native_call(backward_code)
        self.assertEqual(loss(values.detach().clone().requires_grad_()), actual,
                         atol=1e-3 if dtype == torch.float16 else 1e-5,
                         rtol=1e-3 if dtype == torch.float16 else 1.3e-5)

    @parametrize("case", ["extreme", "empty", "scalar"])
    @torch._inductor.config.patch({
        "test_configs.runtime_triton_shape_assert": True,
        "test_configs.runtime_triton_dtype_assert": True,
        "force_disable_caches": True,
        "implicit_fallbacks": False,
    })
    def test_shape_and_range_regressions(self, device, case):
        # Shapes the fused decomposition must keep handling: the abs-free
        # log_sigmoid over saturating inputs, a zero-element reduction, and a 0-d
        # input (the degenerate kernel whose scalar loads keep rank 0).
        if case == "extreme":
            values = torch.tensor([[-60.0, -12.0, 0.0], [7.0, 24.0, 60.0]])
            reduction = "mean"
        elif case == "empty":
            values = torch.zeros(0, 4)
            reduction = "sum"
        else:
            values = torch.tensor(0.3)
            reduction = "mean"
        targets = torch.full_like(values, 0.7)
        dtype = torch.float16
        device_values = values.to(dtype).to(device).requires_grad_()
        device_targets = targets.to(dtype).to(device)
        reference = values.double().requires_grad_()

        def loss(x, t):
            return torch.nn.functional.binary_cross_entropy_with_logits(x, t, reduction=reduction)

        eager = loss(device_values, device_targets)
        actual = torch.compile(loss, fullgraph=True)(device_values, device_targets)
        expected = loss(reference, targets.double())
        self.assertEqual(actual, eager, atol=1e-3, rtol=1e-3)
        # NPU has no float64, so .double() there silently stays float32: move to
        # the host first, otherwise the comparison against the CPU float64
        # reference fails on dtype instead of on values.
        self.assertEqual(actual.cpu().double(), expected, atol=1e-2, rtol=1e-2)
        if values.numel():
            grad = torch.ones_like(actual)
            compiled_grad = torch.autograd.grad(actual, device_values, grad, retain_graph=True)[0]
            eager_grad = torch.autograd.grad(eager, device_values, grad)[0]
            reference_grad = torch.autograd.grad(expected, reference, grad.cpu().double())[0]
            self.assertEqual(compiled_grad, eager_grad, atol=1e-3, rtol=1e-3)
            self.assertEqual(compiled_grad.cpu().double(), reference_grad, atol=1e-2, rtol=1e-2)


@instantiate_parametrized_tests
class TestBCELogitsCPUDecomposition(TestCase):
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("reduction", [0, 1, 2])
    def test_cpu_decomposition_unchanged(self, dtype, reduction):
        from torch._decomp.decompositions import binary_cross_entropy_with_logits
        from torch._inductor.decomposition import decompositions

        values = torch.linspace(-3, 3, 12).reshape(3, 4).to(dtype)
        targets = torch.linspace(0.1, 0.9, 12).reshape(3, 4).to(dtype)
        expected = binary_cross_entropy_with_logits(values, targets, reduction=reduction)
        actual = decompositions[torch.ops.aten.binary_cross_entropy_with_logits.default](
            values, targets, reduction=reduction
        )
        self.assertEqual(actual, expected, atol=0, rtol=0)


instantiate_device_type_tests(TestBCELogitsBackward, globals(), only_for=("npu",))

if __name__ == "__main__":
    run_tests()
