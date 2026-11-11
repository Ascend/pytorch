# Owner(s): ["module: c10d"]
"""Real schema/Fake/autograd checks using a test-only CPU kernel, not HCCL."""
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu
from functional_all_reduce_test_utils import assert_close
from torch_npu.distributed import functional_all_reduce


OP = torch.ops.npu_custom_dist.functional_all_reduce.default


class TestFunctionalAllReduceFramework(TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.is_gloo_available():
            raise unittest.SkipTest("Requires Gloo for the standard SUM backward")
        if dist.is_initialized():
            raise RuntimeError("Run this file in a separate process")
        if torch._C._dispatch_has_kernel_for_dispatch_key("npu_custom_dist::functional_all_reduce", "CPU"):
            raise RuntimeError("Refusing to replace an existing CPU kernel")
        library = torch.library.Library("npu_custom_dist", "IMPL", "CPU")
        cls.library = library
        cls.addClassCleanup(library._destroy)
        # This kernel tests schema/parameter transport, not HCCL reduction semantics.
        library.impl("functional_all_reduce", lambda x, op, group, premul_factor=None:
                     x.clone(memory_format=torch.contiguous_format) if premul_factor is None else x * premul_factor)
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
        cls.addClassCleanup(dist.destroy_process_group)
        cls.group = dist.group.WORLD

    def test_schema_fake_and_effect_registration(self):
        self.assertEqual([str(arg.type) for arg in OP._schema.arguments], ["Tensor", "str", "str", "Optional[float]"])
        self.assertIsNone(OP._schema.arguments[-1].default_value)
        with FakeTensorMode() as mode:
            for dtype in (torch.float32, torch.complex64):
                x = torch.empty(4, 3, dtype=dtype).t()
                y = OP(x, "sum", self.group.group_name)
                self.assertIs(y.fake_mode, mode)
                self.assertEqual((y.shape, y.dtype, y.device), (x.shape, x.dtype, x.device))
                self.assertTrue(y.is_contiguous())
                self.assertFalse(torch._C._is_alias_of(x, y))
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        collective = graph.call_function(OP, (x, "sum", self.group.group_name))
        graph.output(x)
        graph.eliminate_dead_code()
        self.assertIn(collective, graph.nodes)

    def test_scalar_premul_parameter_transport(self):
        x = torch.tensor([2.0, 4.0])
        for factor, expected in ((0.5, [1.0, 2.0]), (0, [0.0, 0.0]), (-2, [-4.0, -8.0])):
            result = functional_all_reduce(x, "PREMUL_SUM", self.group, premul_factor=factor)
            assert_close(funcol.wait_tensor(result), torch.tensor(expected))
        self.assertEqual(x, torch.tensor([2.0, 4.0]))

    def test_reduceop_and_tensor_factors_rejected(self):
        for op in (dist.ReduceOp.SUM, torch_npu.distributed._make_hccl_premul_sum(2.0)):
            with self.assertRaisesRegex(TypeError, "reduceOp must be a string"):
                functional_all_reduce(torch.ones(8), op, self.group)
        for factor in (torch.tensor(2.0), True, "2.0"):
            with self.assertRaisesRegex(TypeError, "premul_factor must be a Python int or float"):
                functional_all_reduce(torch.ones(8), "premul_sum", self.group, premul_factor=factor)

    def test_autograd_registration(self):
        registered = torch._C._dispatch_has_kernel_for_dispatch_key(
            "npu_custom_dist::functional_all_reduce", "Autograd")
        self.assertTrue(registered)

    def test_wait_tensor_identity_backward(self):
        def model(x):
            return funcol.wait_tensor(x)

        self.addCleanup(torch.compiler.reset)
        compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
        for run in (model, compiled):
            for scale in (1.0, 3.0):
                x = torch.arange(8.0, requires_grad=True)
                gradient = torch.arange(1.0, 9.0) * scale
                y = run(x)
                assert_close(y, x)
                y.backward(gradient)
                assert_close(x.grad, gradient)

    def test_compiled_sum_backward_with_explicit_wait(self):
        def model(x):
            return funcol.wait_tensor(functional_all_reduce(x, "sum", self.group))

        self.addCleanup(torch.compiler.reset)
        compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
        for scale in (1.0, 3.0):
            x = torch.arange(8.0, requires_grad=True)
            gradient = torch.arange(1.0, 9.0) * scale
            y = compiled(x)
            assert_close(y, x)
            y.backward(gradient)
            assert_close(x.grad, gradient)

    def test_sum_backward_uses_standard_operator(self):
        x = torch.ones(8, requires_grad=True)
        y = functional_all_reduce(x, "sum", self.group)
        loss = (y * 2).sum()
        standard = torch.ops._c10d_functional.all_reduce
        wait = torch.ops._c10d_functional.wait_tensor
        with patch.object(torch.ops._c10d_functional, "all_reduce", wraps=standard) as calls:
            with patch.object(torch.ops._c10d_functional, "wait_tensor", wraps=wait) as waits:
                with patch.object(torch.ops.npu_custom_dist, "functional_all_reduce",
                                  side_effect=AssertionError("Backward used the new forward path")):
                    loss.backward()
        calls.assert_called_once()
        waits.assert_called_once()
        self.assertEqual(calls.call_args.args[1:], ("sum", self.group.group_name))
        assert_close(x.grad, torch.full_like(x, 2))

    def test_non_sum_backward_rejected_before_communication(self):
        for op in ("avg", "min", "max", "premul_sum"):
            x = torch.ones(8, requires_grad=True)
            y = functional_all_reduce(x, op, self.group, premul_factor=0.5 if op == "premul_sum" else None)
            loss = y.sum()
            with patch.object(torch.ops._c10d_functional, "all_reduce",
                              side_effect=AssertionError("Unexpected backward communication")):
                with self.assertRaisesRegex(RuntimeError, "all_reduce backward only supports 'sum'"):
                    loss.backward()

    def test_compiled_scalar_changes_and_sum_backward(self):
        def premul(x, factor):
            return functional_all_reduce(x, "premul_sum", self.group, premul_factor=factor) + 1

        def train(x):
            return (functional_all_reduce(x, "sum", self.group) * 2).sum()

        self.addCleanup(torch.compiler.reset)
        compiled = torch.compile(premul, backend="aot_eager", fullgraph=True)
        x = torch.tensor([2.0, 4.0])
        for factor, expected in ((0.5, [2.0, 3.0]), (2.0, [5.0, 9.0]), (0.5, [2.0, 3.0])):
            assert_close(compiled(x, factor), torch.tensor(expected))
        compiled_train = torch.compile(train, backend="aot_eager", fullgraph=True)
        x = torch.ones(8, requires_grad=True)
        compiled_train(x).backward()
        assert_close(x.grad, torch.full_like(x, 2))


if __name__ == "__main__":
    run_tests()
