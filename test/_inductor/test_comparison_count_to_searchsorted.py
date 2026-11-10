"""Exact count semantics, match boundaries and NPU compile integration."""

import itertools
import os
import unittest
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_map

from torch_npu._inductor import config
from torch_npu._inductor.fx_passes.ascend_custom_passes.comparison_count_to_searchsorted import (
    _profitable,
    comparison_count_to_searchsorted_pass,
)
from torch_npu._inductor.fx_passes.graph_match_pass import AscendCustomPostPass


aten = torch.ops.aten
PASS_MODULE = (
    "torch_npu._inductor.fx_passes.ascend_custom_passes."
    "comparison_count_to_searchsorted"
)
PASS_COUNTER = "npu_comparison_count_to_searchsorted"
COMPARE_OPS = (torch.ge, torch.gt, torch.le, torch.lt)
FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
VIEW_STYLES = (
    "unsqueeze", "reshape", "expand", "slice", "transpose", "permute", "nested_view"
)


def assert_exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def assert_call_exact(compiled, eager, *args):
    assert_exact(compiled(*args), eager(*args))


def pass_count():
    return counters["inductor"][PASS_COUNTER]


def comparison_count(op, axis, swapped=False, keepdim=False, style="unsqueeze"):
    def fn(q, b):
        qshape = (1, q.numel()) if axis == 0 else (q.numel(), 1)
        bshape = (b.numel(), 1) if axis == 0 else (1, b.numel())
        if style == "unsqueeze":
            qv, bv = q.unsqueeze(axis), b.unsqueeze(1 - axis)
        elif style == "slice":
            qv, bv = (q[None, :], b[:, None]) if axis == 0 else (q[:, None], b[None, :])
        elif style in ("transpose", "permute", "nested_view"):
            qv, bv = q.unsqueeze(1 - axis), b.unsqueeze(axis)
            if style == "transpose":
                qv, bv = qv.transpose(-1, -2), bv.transpose(0, 1)
            else:
                qv, bv = qv.permute(1, 0), bv.permute(-1, -2)
                if style == "nested_view":
                    qv, bv = qv.reshape(qshape), bv.reshape(bshape)
        else:
            qv, bv = q.reshape(qshape), b.reshape(bshape)
            if style == "expand":
                shape = (b.numel(), q.numel()) if axis == 0 else (q.numel(), b.numel())
                qv, bv = qv.expand(shape), bv.expand(shape)
        comparison = op(bv, qv) if swapped else op(qv, bv)
        return comparison.sum(axis, keepdim=keepdim)
    return fn


count_ge = comparison_count(torch.ge, 0)


def float_inputs(dtype):
    limits = torch.finfo(dtype)
    values = torch.tensor(
        [float("nan"), float("-inf"), -limits.max, -1, -0.0, 0.0,
         limits.tiny, limits.tiny / 2, -limits.tiny / 2,
         1, 1 + limits.eps, limits.max, float("inf")], dtype=dtype,
    )
    references = (
        [float("nan"), 1, float("inf"), -0.0, 1, float("-inf"), 0.0],
        [float("nan"), float("nan")],
        [float("inf"), float("inf")],
        [1, -1, 1, 0, limits.tiny / 2, -limits.tiny / 2, 1 + limits.eps],
    )
    return values, tuple(torch.tensor(ref, dtype=dtype) for ref in references)


def float_dynamic_inputs(n, m, dtype, device="cpu"):
    return (
        torch.arange(n, device=device).to(dtype),
        torch.tensor([float("nan"), float("inf"), 0, 1],
                     dtype=dtype, device=device).repeat(m // 4),
    )


def shape_inputs(n, m, dtype=torch.int32):
    return torch.arange(n, dtype=dtype), torch.ones(m, dtype=dtype)


def mapped_rows(boundaries, source):
    positions = torch.arange(378000, device=boundaries.device, dtype=torch.int32)
    count = count_ge(positions, boundaries)
    batch = count.clamp(max=377)
    starts = torch.cat((boundaries.new_zeros(1), boundaries))
    offset = (positions - starts[batch]).clamp(0, 999)
    offset = torch.where(positions < boundaries[-1], offset, 0)
    expanded = batch * 1000 + offset
    # Preserve the downstream division and source mapping from kernel123.
    return (
        source[expanded // 1000] * 1000 + expanded.remainder(1000)
    ).to(torch.int32)


class TestComparisonCountRewrite(unittest.TestCase):
    def graph(
        self, fn=count_ge, args=None, symbolic=False,
        dynamic_values=False,
    ):
        if args is None:
            args = (torch.arange(17, dtype=torch.int32),
                    torch.tensor([5, 2, 2], dtype=torch.int32))
        trace_args = args
        if dynamic_values:
            mode = FakeTensorMode(shape_env=ShapeEnv())
            trace_args = (
                mode.from_tensor(args[0], static_shapes=False),
                mode.from_tensor(args[1], static_shapes=True),
            )
        gm = make_fx(fn, tracing_mode="symbolic" if symbolic else "fake")(*trace_args)
        # Match with NPU metadata; execute semantic fixtures on CPU.
        def retag(t):
            if not isinstance(t, FakeTensor):
                return t
            with torch._C._DisableTorchDispatch():
                meta = torch.empty_strided(
                    t.shape, t.stride(), dtype=t.dtype, device="meta"
                )
            return FakeTensor(t.fake_mode, meta, torch.device("npu:0"))
        for node in gm.graph.nodes:
            if "val" in node.meta:
                node.meta["val"] = tree_map(retag, node.meta["val"])
        return gm, args

    def rewrite(self, gm, force_profitable=True):
        # Semantic fixtures bypass cost policy; profitability tests retain it.
        with patch(
            PASS_MODULE + "._profitable",
            side_effect=(lambda *args: True) if force_profitable else _profitable,
        ):
            comparison_count_to_searchsorted_pass(gm.graph)
        # New scalar factories must follow the CPU execution device.
        for node in gm.graph.nodes:
            if node.target == aten.full.default:
                node.kwargs = {**node.kwargs, "device": torch.device("cpu")}
        gm.graph.lint()
        gm.recompile()
        return sum(n.target == aten.searchsorted.Tensor for n in gm.graph.nodes)

    def check_rewrite(
        self, fn=count_ge, args=None, force_profitable=True, **trace_options
    ):
        gm, args = self.graph(fn, args, **trace_options)
        self.assertEqual(self.rewrite(gm, force_profitable), 1)
        assert_exact(gm(*args), fn(*args))
        return gm

    def test_exact_and_original_boundary_users(self):
        def fn(q, b):
            count = count_ge(q, b)
            return count, b[count.clamp(max=b.numel() - 1)], b[-1]

        cases = ([2, 2, 5], [5, -2, 2, 2], [0, 0, 0], [7], [2147483647, -2147483648, 0])
        q = torch.tensor(
            [-2147483648, -3, -2, 0, 1, 2, 4, 5, 7, 8, 2147483647],
            dtype=torch.int32,
        )
        for values in cases:
            with self.subTest(boundaries=values):
                b = torch.tensor(values, dtype=torch.int32)
                before = b.clone()
                gm = self.check_rewrite(fn, (q, b))
                self.assertTrue(torch.equal(b, before))
                self.assertFalse(
                    any(n.target == aten.ge.Tensor for n in gm.graph.nodes)
                )
                self.assertEqual(self.rewrite(gm), 1)  # Idempotent.

    def test_strided_and_offset_inputs(self):
        q = torch.arange(70, dtype=torch.int32)[3::2]
        b = torch.tensor([99, 5, 88, -2, 77, 5, 66, 20], dtype=torch.int32)[1::2]
        self.check_rewrite(args=(q, b))

    def test_reject_unsupported_forms(self):
        def shared(q, b):
            ge = q[None, :] >= b[:, None]
            return ge.sum(0), ge

        forms = (
            shared,
            lambda q, b: (q[None, :] == b[:, None]).sum(0),
            lambda q, b: (q[None, :] != b[:, None]).sum(0),
            lambda q, b: (q[None, :] >= b[:, None]).sum(0, dtype=torch.int32),
            lambda q, b: (q[None, :] >= b[:, None]).sum(),
            lambda q, b: (q[None, :] >= b[:, None]).sum((0, 1)),
            lambda q, b: (q[None, :] >= b.to(torch.int64)[:, None]).sum(0),
        )
        for fn in forms:
            gm, _ = self.graph(fn)
            self.assertEqual(self.rewrite(gm), 0)
        for dtype in (torch.float64, torch.int16, torch.bool):
            gm, _ = self.graph(
                args=(torch.arange(17).to(dtype), torch.ones(3, dtype=dtype))
            )
            self.assertEqual(self.rewrite(gm), 0)

    def test_float_special_values(self):
        for dtype in FLOAT_DTYPES:
            q, references = float_inputs(dtype)
            for b, op, axis, swapped, keepdim in itertools.product(
                references, COMPARE_OPS, (0, 1), (False, True), (False, True)
            ):
                with self.subTest(dtype=dtype, ref=b.tolist(), op=op, axis=axis,
                                  swapped=swapped, keepdim=keepdim):
                    fn = comparison_count(op, axis, swapped, keepdim)
                    self.check_rewrite(fn, (q, b))

    def test_float_profitability(self):
        for native in (False, True):
            with patch.object(config, "allow_searchsorted_lowering", native):
                self.assertFalse(_profitable(8192, 128, floating=True))
                self.assertFalse(_profitable(32768, 128, floating=True))
                self.assertEqual(_profitable(378000, 378, floating=True), native)
                self.assertTrue(_profitable(32768, 1024, floating=True))

    def test_general_comparison_semantics(self):
        for dtype in (torch.int32, torch.int64):
            limits = torch.iinfo(dtype)
            q = torch.tensor(
                [limits.min, limits.min + 1, -3, 0, 2, 3, limits.max - 1, limits.max],
                dtype=dtype,
            )
            b = torch.tensor([limits.max, 2, limits.min, 2, -3], dtype=dtype)
            for op, axis, swapped, keepdim, style in itertools.product(
                COMPARE_OPS, (0, 1),
                (False, True), (False, True),
                VIEW_STYLES,
            ):
                with self.subTest(dtype=dtype, op=op, axis=axis, swapped=swapped,
                                  keepdim=keepdim, style=style):
                    fn = comparison_count(op, axis, swapped, keepdim, style)
                    gm = self.check_rewrite(fn, (q, b))
                    for node in gm.graph.nodes:
                        if "tensor_meta" in node.meta:
                            self.assertEqual(
                                node.meta["tensor_meta"].shape,
                                node.meta["val"].shape,
                            )

    def test_profitability_gate(self):
        for n, m, expected in (
            (17, 3, 0), (8192, 31, 0), (255, 4096, 0),
            (2047, 128, 0), (4096, 128, 1), (378000, 378, 1),
        ):
            with self.subTest(n=n, m=m):
                gm, _ = self.graph(args=shape_inputs(n, m))
                self.assertEqual(self.rewrite(gm, force_profitable=False), expected)
        for n, m, trace, force in (
            (0, 3, {}, True), (17, 0, {}, True),
            (512, 31, {"symbolic": True}, False),
            (512, 65537, {"symbolic": True}, False),
            (17, 128, {"dynamic_values": True}, False),
        ):
            with self.subTest(n=n, m=m, trace=trace):
                gm, _ = self.graph(args=shape_inputs(n, m), **trace)
                self.assertEqual(self.rewrite(gm, force), 0)

    def test_dynamic_values_static_references(self):
        for style in ("unsqueeze", "slice", "expand", "transpose"):
            fn = comparison_count(torch.le, 1, keepdim=True, style=style)
            reference = torch.tensor([5, -2, 5, 0], dtype=torch.int64).repeat(32)
            args = (torch.arange(4096, dtype=torch.int64), reference)
            gm = self.check_rewrite(
                fn, args, dynamic_values=True, force_profitable=False,
            )
            value = next(n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder")
            self.assertIsInstance(value.shape[0], torch.SymInt)
            for size in (4096, 8192):
                inputs = (torch.arange(size, dtype=torch.int64), args[1])
                assert_exact(gm(*inputs), fn(*inputs))

    def test_unbacked_value_length_skips(self):
        shape_env = ShapeEnv()
        value_count = shape_env.create_unbacked_symint()
        self.assertFalse(_profitable(value_count, 128))
        self.assertFalse(shape_env.guards)
        reference_count = shape_env.create_unbacked_symint()
        self.assertFalse(_profitable(8192, reference_count))
        self.assertFalse(shape_env.guards)

    def test_cache_fingerprint(self):
        def fingerprint(native):
            with patch.object(config, "allow_searchsorted_lowering", native):
                return AscendCustomPostPass().uuid()

        self.assertNotEqual(fingerprint(False), fingerprint(True))


@unittest.skipUnless(
    os.getenv("COMPARISON_COUNT_RUN_NPU_TESTS") == "1", "opt-in NPU integration"
)
class TestComparisonCountCompile(unittest.TestCase):
    def setUp(self):
        torch.npu.set_device(0)
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def compile_tracked(self, fn, dynamic=True):
        from torch._inductor.compile_fx import compile_fx

        builds = []

        def backend(gm, inputs):
            before = pass_count()
            result = compile_fx(gm, inputs)
            builds.append(pass_count() - before)
            return result

        return torch.compile(
            fn, backend=backend, fullgraph=True, dynamic=dynamic
        ), builds

    def compile_exact(self, fn, args):
        before = pass_count()
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        assert_call_exact(compiled, fn, *args)
        self.assertGreater(pass_count(), before)
        return compiled

    def check_dynamic(self, fn, inputs, expected_builds=(1,), cpu_golden=False):
        compiled, builds = self.compile_tracked(fn)
        for args in inputs:
            actual = compiled(*args)
            assert_exact(actual, fn(*args))
            if cpu_golden:
                assert_exact(actual.cpu(), fn(*(t.cpu() for t in args)))
        self.assertEqual(builds, list(expected_builds))

    def test_count(self):
        # Include adjacent large integers: sorting must not lose precision by
        # internally converting INT32 values to FLOAT.
        q = torch.tensor(
            [-2147483648, -1, 0, 16777216, 16777217, 2147483646, 2147483647],
            dtype=torch.int32, device="npu",
        ).repeat(512)
        b = torch.tensor(
            [2147483647, 16777217, -2147483648, 0, 2147483646, 16777216],
            dtype=torch.int32, device="npu",
        ).repeat(32)
        torch._dynamo.reset()
        self.compile_exact(count_ge, (q, b))

    def test_float_special_values(self):
        for dtype in FLOAT_DTYPES:
            q, references = float_inputs(dtype)
            q = q.repeat(512)
            device_q = q.npu()
            for op, axis, swapped in itertools.product(
                COMPARE_OPS, (0, 1), (False, True)
            ):
                torch._dynamo.reset()
                fn = comparison_count(op, axis, swapped, keepdim=True)
                compiled, builds = self.compile_tracked(fn, dynamic=False)
                # Bypass only cost policy; reuse both reference-length graphs
                # across changed contents, including all-NaN and all-inf inputs.
                with patch(PASS_MODULE + "._profitable", return_value=True):
                    for ref in references:
                        with self.subTest(dtype=dtype, ref=ref.tolist(), op=op,
                                          axis=axis, swapped=swapped):
                            b = ref.repeat(64)
                            inputs = device_q, b.npu()
                            actual = compiled(*inputs)
                            assert_exact(actual, fn(*inputs))
                            assert_exact(actual.cpu(), fn(q, b))
                self.assertEqual(builds, [1, 1])

    def test_float_dynamic_shapes(self):
        fn = comparison_count(torch.lt, 1, swapped=True, keepdim=True)
        for dtype in FLOAT_DTYPES:
            with self.subTest(dtype=dtype):
                torch._dynamo.reset()
                inputs = (float_dynamic_inputs(n, m, dtype, "npu")
                          for n, m in ((32768, 1024), (40000, 2048)))
                self.check_dynamic(fn, inputs, cpu_golden=True)

    def test_general_forms(self):
        cases = (
            (torch.ge, 1, False, True, "reshape", torch.int32),
            (torch.le, 0, True, False, "expand", torch.int32),
            (torch.gt, 1, False, True, "slice", torch.int64),
            (torch.lt, 0, False, False, "unsqueeze", torch.int64),
        )
        for op, axis, swapped, keepdim, style, dtype in cases:
            with self.subTest(op=op, axis=axis, swapped=swapped, dtype=dtype):
                torch._dynamo.reset()
                limits = torch.iinfo(dtype)
                q = torch.tensor(
                    [limits.min, limits.min + 1, -1, 0, 2,
                     limits.max - 1, limits.max], device="npu", dtype=dtype,
                ).repeat(512)
                b = torch.tensor([2, limits.max, limits.min, 2, 0, -1],
                                 device="npu", dtype=dtype).repeat(32)
                fn = comparison_count(op, axis, swapped, keepdim, style)
                self.compile_exact(fn, (q, b))

    def test_dynamic_reuse_and_profitability_ranges(self):
        reference = torch.tensor([5, -2, 5, 0], device="npu",
                                 dtype=torch.int32).repeat(32)
        torch._dynamo.mark_static(reference, 0)
        inputs = ((torch.arange(n, device="npu", dtype=torch.int32), reference)
                  for n in (4096, 8192, 3072, 17, 31, 0, 6144))
        # One profitable range, one skipped range, and a zero-size specialization.
        self.check_dynamic(count_ge, inputs, expected_builds=(1, 0, 0))

    def test_both_lengths_dynamic_compile(self):
        for op, dtype, swapped in (
            (torch.ge, torch.int32, False),
            (torch.gt, torch.int64, False),
            (torch.le, torch.int32, False),
            (torch.lt, torch.int64, False),
            (torch.lt, torch.int64, True),
        ):
            with self.subTest(op=op, dtype=dtype, swapped=swapped):
                torch._dynamo.reset()
                fn = comparison_count(
                    op, 1, swapped=swapped, keepdim=True, style="transpose"
                )
                compiled, builds = self.compile_tracked(fn)
                limits = torch.iinfo(dtype)
                for n, m in ((32768, 128), (65536, 256), (49152, 96)):
                    values = torch.tensor(
                        [limits.min, -1, 0, 5, limits.max],
                        device="npu", dtype=dtype,
                    ).repeat((n + 4) // 5)[:n].clone()
                    reference = torch.tensor(
                        [limits.max, 5, limits.min, 5], device="npu", dtype=dtype
                    ).repeat(m // 4)
                    assert_call_exact(compiled, fn, values, reference)
                self.assertEqual(builds, [1])
                for n, m in (
                    (8192, 128), (17, 31), (4096, 0), (0, 128), (65536, 256)
                ):
                    values = torch.arange(n, dtype=dtype, device="npu")
                    reference = torch.arange(m, dtype=dtype, device="npu")
                    assert_call_exact(compiled, fn, values, reference)
                self.assertEqual(builds, [1, 0, 0, 0])

    def test_full_fragment(self):
        source = torch.arange(2, dtype=torch.int64, device="npu").repeat_interleave(189)
        compiled = None
        for lengths in ((317, 821), (0, 1000), (0, 0), (1000, 1000)):
            lens = torch.tensor(lengths, dtype=torch.int32, device="npu")
            boundaries = lens[source].cumsum(0, dtype=torch.int32)
            if compiled is None:
                compiled = self.compile_exact(mapped_rows, (boundaries, source))
            else:
                assert_call_exact(compiled, mapped_rows, boundaries, source)

    def check_training(self, dynamic):
        class CountIndexedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Dyadic weights make outputs and duplicate-index gradients exact.
                labels = torch.arange(1024) // 32 + 1
                initial = labels[:, None].expand(1024, 4).float() / 32
                self.weight = torch.nn.Parameter(initial.npu())

            def forward(self, values, reference):
                # Retain pass workload but bound backward atomic contention.
                index = count_ge(values, reference)[::256].clamp(max=1023)
                output = torch.index_select(self.weight, 0, index)
                return output, output.square().sum()

        eager = CountIndexedModel().train()
        model = CountIndexedModel().train()
        torch._dynamo.mark_static(model.weight)
        initial = model.weight.detach().clone()
        compiled, builds = self.compile_tracked(model, dynamic=dynamic)
        before = pass_count()
        shapes = ((32768, 128), (32768, 256), (65536, 128)) if dynamic else (
            (8192, 128),
        )
        for size, reference_count in shapes:
            with torch.no_grad():
                eager.weight.copy_(initial)
                model.weight.copy_(initial)
            eager.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            values = torch.tensor(
                [-2147483648, -4, -3, -1, 0, 5, 7, 27, 2147483647],
                dtype=torch.int32, device="npu",
            ).repeat((size + 8) // 9)[:size].clone()
            reference = torch.tensor(
                [27, 0, -3, 0], dtype=torch.int32, device="npu"
            ).repeat(reference_count // 4)
            self.assertTrue(torch.is_grad_enabled())
            actual, actual_loss = compiled(values, reference)
            expected, expected_loss = eager(values, reference)
            assert_exact((actual, actual_loss), (expected, expected_loss))
            actual_loss.backward()
            expected_loss.backward()
            assert_exact(model.weight.grad, eager.weight.grad)
            # The dyadic learning rate also makes parameter updates exact.
            with torch.no_grad():
                model.weight.add_(model.weight.grad, alpha=-0.25)
                eager.weight.add_(eager.weight.grad, alpha=-0.25)
            assert_exact(model.weight, eager.weight)
        self.assertEqual(builds, [1])
        self.assertEqual(pass_count() - before, 1)

    def test_training_forward_backward(self):
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                torch._dynamo.reset()
                self.check_training(dynamic)


if __name__ == "__main__":
    unittest.main()
