import unittest

import torch
import torch.utils._pytree as pytree
from torch_npu.npu._npugraph_conditional_nodes import NPUGraphCaptureControlFlowOpDispatchMode

def _clone_leaf(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    return x


def _pred_to_bool(pred):
    return bool(pred.detach().cpu().item())


@unittest.skip("Conditional ModelRI RTS interfaces are unavailable in the current UT environment")
class TestACLGraphConditionalNodes(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("requires npu")
        torch.npu.set_device(0)

    def assertTreeEqual(self, actual, expected):
        actual_flat, actual_spec = pytree.tree_flatten(actual)
        expected_flat, expected_spec = pytree.tree_flatten(expected)
        self.assertEqual(actual_spec, expected_spec)
        self.assertEqual(len(actual_flat), len(expected_flat))
        for actual_leaf, expected_leaf in zip(actual_flat, expected_flat):
            if isinstance(actual_leaf, torch.Tensor):
                torch.testing.assert_close(
                    actual_leaf.detach().cpu(),
                    expected_leaf.detach().cpu(),
                    rtol=1e-4,
                    atol=1e-4,
                )
            else:
                self.assertEqual(actual_leaf, expected_leaf)

    def check_bare_aclgraph(self, fn, args, expected_fn):
        side_stream = torch.npu.Stream()
        graph = torch.npu.NPUGraph()

        with (
            torch.npu.graph(graph, stream=side_stream),
            NPUGraphCaptureControlFlowOpDispatchMode(),
        ):
            captured_output = fn(*args)

        torch.npu.synchronize()
        eager_output = pytree.tree_map(_clone_leaf, expected_fn(*args))

        graph.replay()
        torch.npu.synchronize()
        replay_output = pytree.tree_map(_clone_leaf, captured_output)

        self.assertTreeEqual(replay_output, eager_output)

    def test_cond_not_nested_true_and_false(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        def fn(x, pred):
            return torch.cond(pred, true_fn, false_fn, (x,))

        def expected_fn(x, pred):
            return true_fn(x) if _pred_to_bool(pred) else false_fn(x)

        x = torch.randn(4, device="npu")
        self.check_bare_aclgraph(fn, (x, torch.tensor(True, device="npu")), expected_fn)
        self.check_bare_aclgraph(fn, (x, torch.tensor(False, device="npu")), expected_fn)

    def test_cond_multi_output(self):
        def true_fn(x):
            return x.sin(), x + 1

        def false_fn(x):
            return x.cos(), x - 1

        def fn(x, pred):
            y0, y1 = torch.cond(pred, true_fn, false_fn, (x,))
            return y0 + y1

        def expected_fn(x, pred):
            y0, y1 = true_fn(x) if _pred_to_bool(pred) else false_fn(x)
            return y0 + y1

        x = torch.randn(4, 3, device="npu")
        self.check_bare_aclgraph(fn, (x, torch.tensor(True, device="npu")), expected_fn)
        self.check_bare_aclgraph(fn, (x, torch.tensor(False, device="npu")), expected_fn)

    def test_cond_triply_nested(self):
        def level3_true(x):
            return x.sin()

        def level3_false(x):
            return x.cos()

        def level2_true(x, p2):
            return torch.cond(p2, level3_true, level3_false, (x,))

        def level2_false(x, p2):
            return torch.cond(p2, lambda t: t + 1, lambda t: t - 1, (x,))

        def level1_true(x, p1, p2):
            return torch.cond(p1, level2_true, level2_false, (x, p2))

        def level1_false(x, p1, p2):
            return torch.cond(p1, level2_false, level2_true, (x, p2))

        def fn(x, p0, p1, p2):
            return torch.cond(p0, level1_true, level1_false, (x, p1, p2))

        def expected_level2_true(x, p2):
            return level3_true(x) if _pred_to_bool(p2) else level3_false(x)

        def expected_level2_false(x, p2):
            return x + 1 if _pred_to_bool(p2) else x - 1

        def expected_level1_true(x, p1, p2):
            return expected_level2_true(x, p2) if _pred_to_bool(p1) else expected_level2_false(x, p2)

        def expected_level1_false(x, p1, p2):
            return expected_level2_false(x, p2) if _pred_to_bool(p1) else expected_level2_true(x, p2)

        def expected_fn(x, p0, p1, p2):
            return expected_level1_true(x, p1, p2) if _pred_to_bool(p0) else expected_level1_false(x, p1, p2)

        x = torch.randn(4, device="npu")
        self.check_bare_aclgraph(
            fn,
            (
                x,
                torch.tensor(True, device="npu"),
                torch.tensor(True, device="npu"),
                torch.tensor(False, device="npu"),
            ),
            expected_fn,
        )
        self.check_bare_aclgraph(
            fn,
            (
                x,
                torch.tensor(False, device="npu"),
                torch.tensor(False, device="npu"),
                torch.tensor(True, device="npu"),
            ),
            expected_fn,
        )

    def test_cond_followed_by_consumer(self):
        def true_fn(x):
            return x * x

        def false_fn(x):
            return x + x

        def fn(x, pred):
            y = torch.cond(pred, true_fn, false_fn, (x,))
            return y + x

        def expected_fn(x, pred):
            y = true_fn(x) if _pred_to_bool(pred) else false_fn(x)
            return y + x

        x = torch.randn(4, device="npu")
        self.check_bare_aclgraph(fn, (x, torch.tensor(True, device="npu")), expected_fn)
        self.check_bare_aclgraph(fn, (x, torch.tensor(False, device="npu")), expected_fn)

    def test_cond_two_independent_nodes(self):
        def true_fn(x):
            return x * x

        def false_fn(x):
            return x + x

        def fn(x, pred1, pred2):
            y0 = torch.cond(pred1, true_fn, false_fn, (x,))
            y1 = torch.cond(pred2, false_fn, true_fn, (x,))
            return y0 + y1

        def expected_fn(x, pred1, pred2):
            y0 = true_fn(x) if _pred_to_bool(pred1) else false_fn(x)
            y1 = false_fn(x) if _pred_to_bool(pred2) else true_fn(x)
            return y0 + y1

        x = torch.randn(4, device="npu")
        self.check_bare_aclgraph(
            fn,
            (
                x,
                torch.tensor(True, device="npu"),
                torch.tensor(False, device="npu"),
            ),
            expected_fn,
        )

    def test_cond_no_operands(self):
        def true_fn():
            return torch.ones(4, device="npu")

        def false_fn():
            return torch.zeros(4, device="npu")

        def fn(pred):
            return torch.cond(pred, true_fn, false_fn, ())

        def expected_fn(pred):
            return true_fn() if _pred_to_bool(pred) else false_fn()

        self.check_bare_aclgraph(fn, (torch.tensor(True, device="npu"),), expected_fn)
        self.check_bare_aclgraph(fn, (torch.tensor(False, device="npu"),), expected_fn)

    def test_cond_functionalized_branch_mutation(self):
        def true_fn(x):
            y = x.sin()
            y.add_(4)
            return x.sin().max() + y.sum()

        def false_fn(x):
            return x.cos().min()

        def f(x):
            pred = x.shape[0] == 1
            return torch.cond(pred, true_fn, false_fn, (x,))

        def fn(x, pred):
            return torch.cond(pred, true_fn, false_fn, (x,))

        x = torch.ones(4, 5, device="npu")
        functional_f = torch.func.functionalize(f)
        self.assertTreeEqual(functional_f(x), f(x))

        def expected_fn(x, pred):
            return fn(x, pred)

        pred = torch.tensor(x.shape[0] == 1, device="npu")
        self.check_bare_aclgraph(fn, (x, pred), expected_fn)

    def test_cond_nn_module_forward_backward_parameter_grads(self):
        class CondLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.true_linear = torch.nn.Linear(4, 4)
                self.false_linear = torch.nn.Linear(4, 4)

            def forward(self, x, pred):
                def true_fn(x):
                    return self.true_linear(x).sin()

                def false_fn(x):
                    return self.false_linear(x).cos()

                return torch.cond(pred, true_fn, false_fn, (x,))

        def clone_model(model):
            cloned_model = CondLinear().npu()
            cloned_model.load_state_dict(model.state_dict())
            return cloned_model

        def run_eager(model, x, pred):
            model.zero_grad(set_to_none=True)
            x = x.detach().clone().requires_grad_(True)
            out = model(x, pred)
            loss = out.sum()
            loss.backward()
            return (
                out.detach(),
                x.grad.detach(),
                [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()],
            )

        def run_aclgraph(model, x, pred):
            model.zero_grad(set_to_none=True)
            x = x.detach().clone().requires_grad_(True)
            side_stream = torch.npu.Stream()
            graph = torch.npu.NPUGraph()

            with (
                torch.npu.graph(graph, stream=side_stream),
                NPUGraphCaptureControlFlowOpDispatchMode(),
            ):
                captured_out = model(x, pred)

            graph.replay()
            torch.npu.synchronize()
            loss = captured_out.sum()
            loss.backward()
            return (
                captured_out.detach(),
                x.grad.detach(),
                [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()],
            )

        base_model = CondLinear().npu()
        x = torch.randn(3, 4, device="npu")

        for pred_value in (True, False):
            pred = torch.tensor(pred_value, device="npu")
            eager_model = clone_model(base_model)
            graph_model = clone_model(base_model)

            eager_out, eager_x_grad, eager_param_grads = run_eager(eager_model, x, pred)
            graph_out, graph_x_grad, graph_param_grads = run_aclgraph(graph_model, x, pred)

            torch.testing.assert_close(graph_out.detach().cpu(), eager_out.detach().cpu(), rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(graph_x_grad.detach().cpu(), eager_x_grad.detach().cpu(), rtol=1e-4, atol=1e-4)
            for graph_grad, eager_grad in zip(graph_param_grads, eager_param_grads):
                if eager_grad is None:
                    self.assertIsNone(graph_grad)
                else:
                    torch.testing.assert_close(graph_grad.detach().cpu(), eager_grad.detach().cpu(), rtol=1e-4, atol=1e-4)

    def test_cond_rng_inside_branch_errors(self):
        def true_fn(x):
            return x + torch.randn_like(x)

        def false_fn(x):
            return x * 2

        def fn(x, pred):
            return torch.cond(pred, true_fn, false_fn, (x,))

        x = torch.ones(4, device="npu")
        pred = torch.tensor(True, device="npu")
        graph = torch.npu.NPUGraph()
        side_stream = torch.npu.Stream()

        with self.assertRaisesRegex(
            RuntimeError,
            "RNG op during graph capture but generator is not registered",
        ):
            with (
                torch.npu.graph(graph, stream=side_stream),
                NPUGraphCaptureControlFlowOpDispatchMode(),
            ):
                fn(x, pred)

    def test_rng_outside_conditional_node_does_not_error(self):
        def true_fn(x):
            return x * 2

        def false_fn(x):
            return x * 3

        def fn(x, pred):
            y = torch.cond(pred, true_fn, false_fn, (x,))
            return y + torch.randn_like(y)

        def expected_fn(x, pred):
            # RNG is outside the conditional node, so only verify shape and dtype
            # through the common replay path below.
            y = true_fn(x) if _pred_to_bool(pred) else false_fn(x)
            return torch.empty_like(y)

        x = torch.ones(4, device="npu")
        side_stream = torch.npu.Stream()
        graph = torch.npu.NPUGraph()
        pred = torch.tensor(True, device="npu")

        with (
            torch.npu.graph(graph, stream=side_stream),
            NPUGraphCaptureControlFlowOpDispatchMode(),
        ):
            captured_output = fn(x, pred)

        graph.replay()
        torch.npu.synchronize()

        expected = expected_fn(x, pred)
        self.assertEqual(captured_output.shape, expected.shape)
        self.assertEqual(captured_output.dtype, expected.dtype)

    def test_cond_branch_allocations_replay(self):
        def true_fn(x):
            y = torch.empty_like(x)
            y.copy_(x + 1)
            return y

        def false_fn(x):
            y = torch.empty_like(x)
            y.copy_(x - 1)
            return y

        def fn(x, pred):
            return torch.cond(pred, true_fn, false_fn, (x,))

        def expected_fn(x, pred):
            return true_fn(x) if _pred_to_bool(pred) else false_fn(x)

        x = torch.randn(4, 3, device="npu")
        self.check_bare_aclgraph(fn, (x, torch.tensor(True, device="npu")), expected_fn)
        self.check_bare_aclgraph(fn, (x, torch.tensor(False, device="npu")), expected_fn)

    def test_cond_branch_multi_output_allocations_replay(self):
        def true_fn(x):
            y0 = torch.empty_like(x)
            y1 = torch.empty_like(x)
            y0.copy_(x.sin())
            y1.copy_(x + 1)
            return y0, y1

        def false_fn(x):
            y0 = torch.empty_like(x)
            y1 = torch.empty_like(x)
            y0.copy_(x.cos())
            y1.copy_(x - 1)
            return y0, y1

        def fn(x, pred):
            y0, y1 = torch.cond(pred, true_fn, false_fn, (x,))
            return y0 + y1

        def expected_fn(x, pred):
            y0, y1 = true_fn(x) if _pred_to_bool(pred) else false_fn(x)
            return y0 + y1

        x = torch.randn(4, 3, device="npu")
        self.check_bare_aclgraph(fn, (x, torch.tensor(True, device="npu")), expected_fn)
        self.check_bare_aclgraph(fn, (x, torch.tensor(False, device="npu")), expected_fn)

    def test_cond_no_operand_multi_output_allocations_replay(self):
        def true_fn():
            return torch.zeros(8, device="npu"), torch.zeros(8, device="npu")

        def false_fn():
            return torch.ones(8, device="npu"), torch.ones(8, device="npu")

        def fn(pred):
            y0, y1 = torch.cond(pred, true_fn, false_fn, ())
            return y0 + y1

        def expected_fn(pred):
            y0, y1 = true_fn() if _pred_to_bool(pred) else false_fn()
            return y0 + y1

        self.check_bare_aclgraph(fn, (torch.tensor(True, device="npu"),), expected_fn)
        self.check_bare_aclgraph(fn, (torch.tensor(False, device="npu"),), expected_fn)

    def test_nested_cond_branch_allocations_replay(self):
        def inner_true(x):
            y = torch.empty_like(x)
            y.copy_(x.sin())
            return y

        def inner_false(x):
            y = torch.empty_like(x)
            y.copy_(x.cos())
            return y

        def outer_true(x, inner_pred):
            y = torch.empty_like(x)
            y.copy_(x + 1)
            return torch.cond(inner_pred, inner_true, inner_false, (y,))

        def outer_false(x, inner_pred):
            y = torch.empty_like(x)
            y.copy_(x - 1)
            return torch.cond(inner_pred, inner_false, inner_true, (y,))

        def fn(x, outer_pred, inner_pred):
            return torch.cond(outer_pred, outer_true, outer_false, (x, inner_pred))

        def expected_outer_true(x, inner_pred):
            y = torch.empty_like(x)
            y.copy_(x + 1)
            return inner_true(y) if _pred_to_bool(inner_pred) else inner_false(y)

        def expected_outer_false(x, inner_pred):
            y = torch.empty_like(x)
            y.copy_(x - 1)
            return inner_false(y) if _pred_to_bool(inner_pred) else inner_true(y)

        def expected_fn(x, outer_pred, inner_pred):
            return expected_outer_true(x, inner_pred) if _pred_to_bool(outer_pred) else expected_outer_false(x, inner_pred)

        x = torch.randn(4, 3, device="npu")
        self.check_bare_aclgraph(
            fn,
            (
                x,
                torch.tensor(True, device="npu"),
                torch.tensor(False, device="npu"),
            ),
            expected_fn,
        )
        self.check_bare_aclgraph(
            fn,
            (
                x,
                torch.tensor(False, device="npu"),
                torch.tensor(True, device="npu"),
            ),
            expected_fn,
        )

    def test_replay_uses_updated_predicate_tensor(self):
        def true_fn(x):
            return x + 10

        def false_fn(x):
            return x - 10

        def fn(x, pred):
            return torch.cond(pred, true_fn, false_fn, (x,))

        x = torch.randn(4, device="npu")
        pred = torch.tensor(True, device="npu")
        side_stream = torch.npu.Stream()
        graph = torch.npu.NPUGraph()

        with (
            torch.npu.graph(graph, stream=side_stream),
            NPUGraphCaptureControlFlowOpDispatchMode(),
        ):
            captured_output = fn(x, pred)

        pred.fill_(True)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured_output.detach().cpu(), (x + 10).detach().cpu())

        pred.fill_(False)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured_output.detach().cpu(), (x - 10).detach().cpu())

    def test_cond_requires_npu_predicate(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        x = torch.randn(4, device="npu")
        graph = torch.npu.NPUGraph()

        with self.assertRaisesRegex(ValueError, "Conditions must be on an npu device"):
            with (
                torch.npu.graph(graph),
                NPUGraphCaptureControlFlowOpDispatchMode(),
            ):
                torch.cond(torch.tensor(True), true_fn, false_fn, (x,))

    def test_cond_requires_bool_predicate(self):
        def true_fn(x):
            return x + 1

        def false_fn(x):
            return x - 1

        x = torch.randn(4, device="npu")
        pred = torch.tensor(1, device="npu", dtype=torch.int32)
        graph = torch.npu.NPUGraph()

        with self.assertRaisesRegex(RuntimeError, "Conditions must be bool tensors"):
            with (
                torch.npu.graph(graph),
                NPUGraphCaptureControlFlowOpDispatchMode(),
            ):
                torch.cond(pred, true_fn, false_fn, (x,))


if __name__ == "__main__":
    unittest.main()
