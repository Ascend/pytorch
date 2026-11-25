import unittest

import torch
import torch.utils._pytree as pytree
from torch_npu._compat.version import CURRENT_VERSION
from torch_npu.npu._npugraph_conditional_nodes import (
    ControlFlowOpWarmupDispatchMode,
    NPUGraphCaptureControlFlowOpDispatchMode,
)

def _clone_leaf(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    return x


def _pred_to_bool(pred):
    return bool(pred.detach().cpu().item())


_COND_CAPTURE_SUPPORTED = None


def _conditional_capture_supported():
    """Probe once whether the driver and firmware support conditional model capture.

    A failed probe is contained: the conditional handle is created before the
    child capture begins, so a failure leaves no active capture behind.
    """
    global _COND_CAPTURE_SUPPORTED
    if _COND_CAPTURE_SUPPORTED is None:
        _COND_CAPTURE_SUPPORTED = False
        probe = torch.npu.NPUGraph()
        try:
            pred = torch.zeros((), device="npu", dtype=torch.bool)
            scalar = torch.zeros((), device="npu")
            with torch.npu.graph(probe):
                probe.begin_capture_to_if_node(pred)
                scalar.add_(1.0)
                probe.end_capture_to_conditional_node()
            torch.npu.synchronize()
            _COND_CAPTURE_SUPPORTED = True
        except RuntimeError:
            pass
        finally:
            probe.reset()
    return _COND_CAPTURE_SUPPORTED


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

    def test_cond_many_if_nodes_across_graphs(self):
        """40 graphs with one conditional node each.

        The stream pool holds 32 streams per device and serves them round-robin,
        and the root capture stream comes from it too, so drawing a body stream
        from the pool wraps around onto a stream that is already capturing and
        AclmdlRICaptureToModelRIBegin fails with 107027.
        Mirrors test_cuda_graph_many_if_nodes_across_graphs of
        pytorch/pytorch#185836.
        """
        pred = torch.zeros((), dtype=torch.bool, device="npu")
        x = torch.zeros((), device="npu")
        for _ in range(40):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                graph.begin_capture_to_if_node(pred)
                x.add_(1.0)
                graph.end_capture_to_conditional_node()

    def test_cond_many_if_nodes_in_one_graph(self):
        """40 conditional nodes inside a single graph.

        Same wrap-around as above, accumulated within one capture instead of
        across captures. Mirrors test_cuda_graph_many_if_nodes_in_one_graph of
        pytorch/pytorch#185836.
        """
        pred = torch.zeros((), dtype=torch.bool, device="npu")
        x = torch.zeros((), device="npu")
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(40):
                graph.begin_capture_to_if_node(pred)
                x.add_(1.0)
                graph.end_capture_to_conditional_node()

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

    def test_cond_five_levels_nested(self):
        """Five levels of nesting must capture and replay.

        Depth only: five bodies plus the root capture draw six streams, far below
        the pool's 32, so this does not exercise the wrap-around. That is what the
        two tests below cover.
        """

        def level5_true(x):
            return x.sin()

        def level5_false(x):
            return x.cos()

        def level4_true(x, p4):
            return torch.cond(p4, level5_true, level5_false, (x,))

        def level4_false(x, p4):
            return torch.cond(p4, lambda t: t * 2, lambda t: t / 2, (x,))

        def level3_true(x, p3, p4):
            return torch.cond(p3, level4_true, level4_false, (x, p4))

        def level3_false(x, p3, p4):
            return torch.cond(p3, level4_false, level4_true, (x, p4))

        def level2_true(x, p2, p3, p4):
            return torch.cond(p2, level3_true, level3_false, (x, p3, p4))

        def level2_false(x, p2, p3, p4):
            return torch.cond(p2, level3_false, level3_true, (x, p3, p4))

        def level1_true(x, p1, p2, p3, p4):
            return torch.cond(p1, level2_true, level2_false, (x, p2, p3, p4))

        def level1_false(x, p1, p2, p3, p4):
            return torch.cond(p1, level2_false, level2_true, (x, p2, p3, p4))

        def fn(x, p0, p1, p2, p3, p4):
            return torch.cond(p0, level1_true, level1_false, (x, p1, p2, p3, p4))

        def expected_level4_true(x, p4):
            return level5_true(x) if _pred_to_bool(p4) else level5_false(x)

        def expected_level4_false(x, p4):
            return x * 2 if _pred_to_bool(p4) else x / 2

        def expected_level3_true(x, p3, p4):
            return expected_level4_true(x, p4) if _pred_to_bool(p3) else expected_level4_false(x, p4)

        def expected_level3_false(x, p3, p4):
            return expected_level4_false(x, p4) if _pred_to_bool(p3) else expected_level4_true(x, p4)

        def expected_level2_true(x, p2, p3, p4):
            return (
                expected_level3_true(x, p3, p4)
                if _pred_to_bool(p2)
                else expected_level3_false(x, p3, p4)
            )

        def expected_level2_false(x, p2, p3, p4):
            return (
                expected_level3_false(x, p3, p4)
                if _pred_to_bool(p2)
                else expected_level3_true(x, p3, p4)
            )

        def expected_level1_true(x, p1, p2, p3, p4):
            return (
                expected_level2_true(x, p2, p3, p4)
                if _pred_to_bool(p1)
                else expected_level2_false(x, p2, p3, p4)
            )

        def expected_level1_false(x, p1, p2, p3, p4):
            return (
                expected_level2_false(x, p2, p3, p4)
                if _pred_to_bool(p1)
                else expected_level2_true(x, p2, p3, p4)
            )

        def expected_fn(x, p0, p1, p2, p3, p4):
            return (
                expected_level1_true(x, p1, p2, p3, p4)
                if _pred_to_bool(p0)
                else expected_level1_false(x, p1, p2, p3, p4)
            )

        def pred(value):
            return torch.tensor(value, device="npu")

        x = torch.randn(4, device="npu")
        self.check_bare_aclgraph(
            fn,
            (x, pred(True), pred(True), pred(True), pred(True), pred(False)),
            expected_fn,
        )
        self.check_bare_aclgraph(
            fn,
            (x, pred(False), pred(False), pred(True), pred(False), pred(True)),
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
            r"RNG within data-dependent conditional nodes is not supported yet\.",
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


@unittest.skipUnless(CURRENT_VERSION >= (2, 14), "NPUGraph while_loop requires PyTorch 2.14 or newer")
@unittest.skip("Conditional ModelRI RTS interfaces are unavailable in the current UT environment")
class TestACLGraphWhileLoop(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("requires NPU")

    def test_condition_update_requires_capture(self):
        graph = torch.npu.NPUGraph()
        with torch.npu.stream(torch.npu.Stream()), self.assertRaisesRegex(RuntimeError, "not capturing"):
            graph.set_conditional_handle_for_current_node(torch.tensor(True, device="npu"))

    def _check_while_replays(self, fn, cases):
        acc = torch.ones((), device="npu", dtype=torch.int32)
        original = acc.cpu()
        limit = torch.zeros((), device=acc.device, dtype=torch.int32)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.stream(stream), ControlFlowOpWarmupDispatchMode():
                fn(acc, limit)
            with torch.npu.graph(graph, stream=stream), NPUGraphCaptureControlFlowOpDispatchMode():
                out_acc = fn(acc, limit)
        except RuntimeError as error:
            if "CondHandleCreate" in str(error) and "207000" in str(error):
                self.skipTest(str(error))
            raise

        torch.npu.current_stream().wait_stream(stream)
        for count, expected_acc in cases:
            with self.subTest(count=count):
                limit.fill_(count)
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(out_acc.cpu(), torch.as_tensor(expected_acc, dtype=acc.dtype))
                torch.testing.assert_close(acc.cpu(), original)
                self.assertEqual(limit.item(), count)

    def test_while_replay_uses_runtime_condition(self):
        def fn(acc, limit):
            def body_fn(value, remaining):
                return value + 1, remaining - 1

            return torch.while_loop(lambda value, remaining: remaining > 0, body_fn, (acc, limit))[0]

        self._check_while_replays(fn, ((3, 4), (0, 1), (5, 6)))

    def test_while_nested_in_while(self):
        def fn(acc, limit):
            def body_fn(value, remaining):
                def inner_body(inner_value, inner_remaining):
                    return inner_value + 2, inner_remaining - 1

                inner_count = torch.full_like(remaining, 2)
                value, _ = torch.while_loop(
                    lambda inner_value, inner_remaining: inner_remaining > 0,
                    inner_body,
                    (value, inner_count),
                )
                return value, remaining - 1

            return torch.while_loop(lambda value, remaining: remaining > 0, body_fn, (acc, limit))[0]

        self._check_while_replays(fn, ((3, 13), (0, 1), (5, 21)))

    def test_while_nested_in_if(self):
        def fn(acc, limit):
            def true_fn(value, count):
                def body_fn(current, remaining):
                    return current + 2, remaining - 1

                return torch.while_loop(lambda current, remaining: remaining > 0, body_fn, (value, count))[0]

            def false_fn(value, count):
                return value - count

            return torch.cond(limit % 2 == 0, true_fn, false_fn, (acc, limit))

        self._check_while_replays(fn, ((2, 5), (3, -2), (0, 1), (4, 9)))

    def test_if_nested_in_while(self):
        def fn(acc, limit):
            def body_fn(value, step):
                updated = torch.cond(step == 0, lambda a: a + 2, lambda a: a + 1, (value,))
                return updated, step + 1

            step = torch.zeros_like(limit)
            return torch.while_loop(lambda value, step: step < limit, body_fn, (acc, step))[0]

        self._check_while_replays(fn, ((3, 5), (0, 1), (5, 7)))


    def test_conditional_body_gets_its_own_save_stream(self):
        import torch_npu
        import torch_npu.npu.graphs as npu_graphs
        from torch_npu.npu._npugraph_conditional_nodes import if_else_node

        shared = npu_graphs._get_save_tensor_stream(0)
        seen = {}

        def record(tag):
            def fn(t):
                seen[tag] = npu_graphs._get_save_tensor_stream(0)
                return t.sin()
            return fn

        x = torch.randn(8, device="npu")
        pred = torch.ones((), dtype=torch.bool, device="npu")
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            if_else_node(pred, record("true"), record("false"), (x,))

        # Each body captures into its own model, so no two of the three streams may be equal.
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(seen["true"].npu_stream, shared.npu_stream)
        self.assertNotEqual(seen["false"].npu_stream, shared.npu_stream)
        self.assertNotEqual(seen["true"].npu_stream, seen["false"].npu_stream)
        # Back outside the bodies the shared stream is in use again.
        self.assertIsNone(torch_npu._C._npu_current_conditional_save_stream())
        g.reset()


class TestACLGraphConditionalNodePool(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("requires npu")
        torch.npu.set_device(0)
        if not _conditional_capture_supported():
            self.skipTest("requires driver and firmware with conditional model capture support")

    def test_conditional_nodes_release_pool(self):
        # Every conditional node re-routes allocations to the graph's pool, which
        # used to leave two extra pool references per node behind so the pool
        # could never be freed after reset().
        # Mirrors pytorch/pytorch#201639.
        for nested in (False, True):
            with self.subTest(nested=nested):
                pred = torch.zeros((), device="npu", dtype=torch.bool)
                x = torch.zeros((), device="npu")
                pool = torch.npu.MemPool()
                self.assertEqual(pool.use_count(), 1)
                g = torch.npu.NPUGraph()
                # capture_begin() takes the pool token, not the MemPool itself.
                with torch.npu.graph(g, pool=pool.id):
                    for _ in range(3):
                        g.begin_capture_to_if_node(pred)
                        x.add_(1.0)
                        if nested:
                            g.begin_capture_to_if_node(pred)
                            x.add_(1.0)
                            g.end_capture_to_conditional_node()
                        g.end_capture_to_conditional_node()
                self.assertEqual(pool.use_count(), 2)
                g.reset()
                self.assertEqual(pool.use_count(), 1)


if __name__ == "__main__":
    unittest.main()
