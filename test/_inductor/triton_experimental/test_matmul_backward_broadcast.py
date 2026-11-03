import os

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, instantiate_parametrized_tests, parametrize, run_tests

import torch_npu
import torch_npu.testing  # noqa: F401
from torch_npu._inductor.decomposition import _matmul_backward_inductor


@instantiate_parametrized_tests
class TestMatmulBackwardGraph(TestCase):
    @parametrize("shared_left", [False, True])
    def test_folded_graph(self, shared_left):
        shapes = ((1, 5, 5), (5, 5, 5)) if shared_left else ((5, 5, 5), (1, 5, 5))
        inputs = tuple(torch.empty(shape) for shape in shapes)
        grad = torch.empty((5, 5, 5))

        def decompose(grad, a, b):
            return _matmul_backward_inductor(grad, a, b, (True, True))

        graph = make_fx(decompose, tracing_mode="fake")(grad, *inputs)
        targets = [node.target for node in graph.graph.nodes if node.op == "call_function"]
        self.assertEqual(targets.count(torch.ops.aten.mm.default), 2)
        self.assertNotIn(torch.ops.aten.bmm.default, targets)

    @parametrize("mask", [(False, False), (True, True)])
    def test_unbacked_batch_guard(self, mask):
        shape_env = ShapeEnv()
        batch = shape_env.create_unbacked_symint()
        with FakeTensorMode(shape_env=shape_env):
            a = torch.empty((batch, 5, 7))
            b = torch.empty((batch, 7, 3))
            grad = torch.empty((batch, 5, 3))
            actual = _matmul_backward_inductor(grad, a, b, mask)
            if any(mask):
                self.assertEqual(actual[0].shape, a.shape)
                self.assertEqual(actual[1].shape, b.shape)
            else:
                self.assertEqual(actual, (None, None))


class TestMatmulBackwardBroadcast(TestCase):
    @dtypes(torch.float16, torch.float32)
    @parametrize("shapes", [
        ((1, 5, 5), (5, 5, 5)),
        ((5, 5, 5), (1, 5, 5)),
        ((1, 1, 5, 5), (1, 5, 5)),
        ((1, 5, 5), (0, 5, 5)),
        ((0, 5, 5), (1, 5, 5)),
    ])
    @parametrize("mask", [(True, True), (True, False), (False, True), (False, False)])
    def test_shared_batch(self, device, dtype, shapes, mask):
        inputs = tuple(torch.randn(shape, device=device, dtype=dtype) for shape in shapes)
        grad = torch.randn_like(inputs[0] @ inputs[1])
        versions = tuple(value._version for value in (*inputs, grad))
        references = tuple(value.cpu().float().requires_grad_() for value in inputs)
        expected = torch.autograd.grad(references[0] @ references[1], references, grad_outputs=grad.cpu().float())
        actual = _matmul_backward_inductor(grad, *inputs, mask)
        expected = tuple(value.to(device=device, dtype=dtype) if enabled else None for value, enabled in zip(expected, mask))
        self.assertEqual(actual, expected)
        self.assertEqual(tuple(value._version for value in (*inputs, grad)), versions)

    @dtypes(torch.float16, torch.float32)
    @parametrize("shared_left", [False, True])
    @torch._inductor.config.patch({"force_disable_caches": True, "implicit_fallbacks": False})
    def test_compiled_shared_batch(self, device, dtype, shared_left):
        shapes = ((1, 5, 5), (5, 5, 5)) if shared_left else ((5, 5, 5), (1, 5, 5))
        inputs = tuple(torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for shape in shapes)

        def fn(a, b):
            return a @ b

        actual = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})(*inputs)
        expected = fn(*inputs)
        grad = torch.randn_like(expected)
        self.assertEqual(actual, expected)
        self.assertEqual(
            torch.autograd.grad(actual, inputs, grad_outputs=grad),
            torch.autograd.grad(expected, inputs, grad_outputs=grad),
        )


instantiate_device_type_tests(TestMatmulBackwardBroadcast, globals(), only_for="npu")

if __name__ == "__main__":
    run_tests()
