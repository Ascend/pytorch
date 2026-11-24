# Owner(s): ["module: inductor"]
import unittest

import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


@instantiate_parametrized_tests
class TestLinalgSolveTriangularMeta(TestCase):
    @staticmethod
    def make_inputs(device, left, upper, batch_case, requires_grad=False):
        # Keep the systems well conditioned; rectangular B distinguishes
        # row-major and column-major output strides.
        a = torch.tensor([[5., .2, .3], [.1, 6., .4], [.2, .3, 7.]])
        a = a.triu() if upper else a.tril()
        b = torch.arange(1, 7, dtype=torch.float32).reshape((3, 2) if left else (2, 3))
        if batch_case == "broadcast_a":
            b = b.unsqueeze(0).repeat(2, 1, 1)
        elif batch_case == "broadcast_b":
            a = a.unsqueeze(0).repeat(2, 1, 1)
        elif batch_case == "broadcast_both":
            a = a.reshape(1, 1, 3, 3).repeat(2, 1, 1, 1)
            b = b.reshape((1, 1) + tuple(b.shape)).repeat(1, 4, 1, 1)
        a = a.to(device).requires_grad_(requires_grad)
        b = b.to(device).requires_grad_(requires_grad)
        return a, b

    @parametrize("device", ["cpu", "npu"])
    @parametrize("left", [False, True])
    @parametrize("upper", [False, True])
    @parametrize("unitriangular", [False, True])
    @parametrize("batch_case", ["none", "broadcast_a", "broadcast_b", "broadcast_both"])
    def test_fake_layout_matches_eager(self, device, left, upper, unitriangular, batch_case):
        if device == "npu" and not torch.npu.is_available():
            self.skipTest("requires NPU")
        a, b = self.make_inputs(device, left, upper, batch_case)
        kwargs = dict(upper=upper, left=left, unitriangular=unitriangular)
        actual = torch.linalg.solve_triangular(a, b, **kwargs)
        # Exercise public dispatch, rather than calling the new Meta function
        # directly, so a missing registration is also caught.
        with FakeTensorMode() as mode:
            fake = torch.linalg.solve_triangular(
                mode.from_tensor(a), mode.from_tensor(b), **kwargs
            )
        expected_batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        self.assertEqual(actual.shape, expected_batch + b.shape[-2:])
        self.assertEqual(fake.shape, actual.shape)
        self.assertEqual(fake.dtype, actual.dtype)
        self.assertEqual(fake.device, actual.device)
        self.assertEqual(fake.stride(), actual.stride())

    @unittest.skipIf(not torch.npu.is_available(), "requires NPU")
    @parametrize("left", [False, True])
    def test_compile_forward_backward(self, left):
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        a, b = self.make_inputs("npu", left, True, "broadcast_a", requires_grad=True)

        def solve(x, y):
            return torch.linalg.solve_triangular(x, y, upper=True, left=left)

        eager = solve(a, b)
        with torch._inductor.config.patch({"size_asserts": True}):
            compiled = torch.compile(solve, fullgraph=True)(a, b)
            grad = torch.arange(1, eager.numel() + 1, device=eager.device, dtype=eager.dtype)
            grad = grad.reshape(eager.shape)
            compiled_grads = torch.autograd.grad(compiled, (a, b), grad)
        eager_grads = torch.autograd.grad(eager, (a, b), grad)
        torch.testing.assert_close(compiled, eager, atol=1e-5, rtol=1e-5)
        self.assertEqual(compiled.stride(), eager.stride())
        for actual, expected in zip(compiled_grads, eager_grads):
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    run_tests()
