import os

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._functorch._aot_autograd.functional_utils import assert_functional_graph
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests

import torch_npu
import torch_npu.testing  # noqa: F401
from torch_npu._inductor.decomposition import _matmul_backward_inductor


class TestMatmulBackwardFunctional(TestCase):
    @dtypes(torch.float16, torch.float32)
    @parametrize("shapes", [
        ((5, 10), (10,)), ((10,), (10, 5)),
        ((5, 5, 0), (0, 5)), ((5, 0), (5, 0, 5)),
        ((5, 5, 0), (0,)), ((0,), (5, 0, 5)),
        ((0, 5, 10), (10, 5)), ((5, 10), (0, 10, 5)),
    ])
    @parametrize("mask", [(True, True), (True, False), (False, True), (False, False)])
    def test_functional_graph(self, device, dtype, shapes, mask):
        a, b = [torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for shape in shapes]
        expected_output = a @ b
        grad = torch.randn_like(expected_output)
        versions = (a._version, b._version, grad._version)

        def decompose(grad, a, b):
            return _matmul_backward_inductor(grad, a, b, mask)

        graph = make_fx(decompose, tracing_mode="fake")(grad, a, b)
        self.assertEqual(assert_functional_graph(graph.graph), 0)
        actual = graph(grad, a, b)
        reference_inputs = tuple(value.detach().cpu().requires_grad_() for value in (a, b))
        expected = torch.autograd.grad(reference_inputs[0] @ reference_inputs[1], reference_inputs, grad_outputs=grad.cpu())
        expected = tuple(value.to(device) for value in expected)
        self.assertEqual(actual, tuple(value if enabled else None for value, enabled in zip(expected, mask)))
        self.assertEqual((a._version, b._version, grad._version), versions)

    @dtypes(torch.float16, torch.float32)
    @parametrize("shapes", [((5, 10), (10,)), ((10,), (10, 5))])
    @parametrize("noncontiguous", [False, True])
    @torch._inductor.config.patch({"force_disable_caches": True, "implicit_fallbacks": False})
    def test_compiled_backward(self, device, dtype, shapes, noncontiguous):
        inputs = []
        for shape in shapes:
            if noncontiguous and len(shape) == 2:
                value = torch.randn(tuple(reversed(shape)), device=device, dtype=dtype).t()
                self.assertFalse(value.is_contiguous())
            else:
                value = torch.randn(shape, device=device, dtype=dtype)
            inputs.append(value.requires_grad_())
        versions = tuple(value._version for value in inputs)

        def fn(a, b):
            return a @ b

        compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})
        actual = compiled(*inputs)
        expected = fn(*inputs)
        grad = torch.randn_like(expected)
        self.assertEqual(actual, expected)
        self.assertEqual(
            torch.autograd.grad(actual, inputs, grad_outputs=grad),
            torch.autograd.grad(expected, inputs, grad_outputs=grad),
        )
        self.assertEqual(tuple(value._version for value in inputs), versions)

    @dtypes(torch.float16, torch.float32)
    @parametrize("shapes", [((5, 5, 0), (0, 5)), ((5, 0), (5, 0, 5))])
    @torch._inductor.config.patch({"force_disable_caches": True, "implicit_fallbacks": False})
    def test_compiled_empty_backward(self, device, dtype, shapes):
        references = tuple(torch.randn(shape, dtype=dtype, requires_grad=True) for shape in shapes)
        inputs = tuple(value.detach().to(device).requires_grad_() for value in references)
        versions = tuple(value._version for value in inputs)

        def fn(a, b):
            return a @ b

        actual = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})(*inputs)
        expected = fn(*references)
        grad = torch.randn_like(expected)
        expected_grads = torch.autograd.grad(expected, references, grad_outputs=grad)
        actual_grads = torch.autograd.grad(actual, inputs, grad_outputs=grad.to(device))
        self.assertEqual(actual, expected.to(device))
        self.assertEqual(actual_grads, tuple(value.to(device) for value in expected_grads))
        self.assertEqual(tuple(value._version for value in inputs), versions)


instantiate_device_type_tests(TestMatmulBackwardFunctional, globals(), only_for="npu")

if __name__ == "__main__":
    run_tests()
