import contextlib
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

import torch_npu
import torch_npu._inductor  # noqa: F401 - initialize the NPU Inductor backend
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    MULTI_SLICE_CONCAT_TARGET,
    fold_cat,
    multi_slice_concat_pass,
    remove_cast_after_slice_by_add_cat as rewrite,
)
from torch_npu._inductor.fx_passes.ascend_custom_passes.register_custom_pass import (
    ASCEND_CUSTOME_PASS_REGISTER,
)

CAT = torch.ops.aten.cat.default
CAST = torch.ops.prims.convert_element_type.default


def pattern(x, y, *, explicit=False, nested=False, dim=1):
    parts = [x[:, i : i + 1] for i in (9, 1, 9, 4, 15)]
    if explicit:
        parts = [p.float() for p in parts]
    if nested:
        parts = [torch.cat(parts, dim)]
    return torch.cat(parts + [y], dim)


def trace(fn, rows=7, dtype=torch.bfloat16, device="npu"):
    with FakeTensorMode():
        x = torch.empty(rows, 19, dtype=dtype, device=device)
        y = torch.empty(rows, 1, device=device)
        gm = make_fx(lambda a, b: fn(a, b))(x, y)
    # make_fx includes identity row slices; post-grad normally removes these.
    for node in list(gm.graph.nodes):
        if node.target == torch.ops.aten.slice.Tensor and node.args[1] == 0:
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
    gm.recompile()
    return gm


def count(gm, target):
    return sum(n.target == target for n in gm.graph.nodes)


class TestGraph(unittest.TestCase):
    def check(self, fn, expected_casts=1, rows=7):
        gm = trace(fn, rows)
        x = torch.randn(rows, 19).bfloat16()
        # Include Inf, NaN and signed zero in selected columns.
        if rows:
            x[0, 9], x[0, 1] = float("inf"), float("nan")
            x[0, 4], x[0, 15] = -0.0, -float("inf")
        y = torch.randn(rows, 1)
        expected = gm(x, y)
        rewrite(gm.graph)
        gm.graph.lint()
        gm.recompile()
        self.assertEqual(count(gm, CAST), expected_casts)
        actual = gm(x, y)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        self.assertTrue(torch.equal(torch.signbit(actual), torch.signbit(expected)))
        code = gm.code
        rewrite(gm.graph)
        gm.recompile()
        self.assertEqual(gm.code, code)
        return gm

    def test_implicit_explicit_nested_and_empty_batch(self):
        for rows in (0, 7, 17):
            for explicit, nested in ((False, False), (True, False), (False, True)):
                with self.subTest(rows=rows, explicit=explicit, nested=nested):
                    self.check(
                        lambda x, y: pattern(
                            x, y, explicit=explicit, nested=nested, dim=-1
                        ),
                        rows=rows,
                    )

    def test_multiple_runs_and_other_users(self):
        def fn(x, y):
            a, b = x[:, 1:2], x[:, 5:7]
            return torch.cat([a, b, y, b, a], 1), a.float(), torch.cat([a, b], 1)

        gm = trace(fn)
        x, y = torch.randn(7, 19).bfloat16(), torch.randn(7, 1)
        expected = gm(x, y)
        rewrite(gm.graph)
        gm.recompile()
        self.assertEqual(count(gm, CAST), 2)
        torch.testing.assert_close(gm(x, y), expected, rtol=0, atol=0)

    def test_misses(self):
        cases = [
            (lambda x, y: pattern(x, y), torch.bfloat16, "cpu"),
            (lambda x, y: torch.cat([x[:, 1:2], y], 1), torch.bfloat16, "npu"),
            (lambda x, y: torch.cat([x[:, 1:2], x[:, 2:3]], 1), torch.bfloat16, "npu"),
            (
                lambda x, y: torch.cat([x[:, 3:3], x[:, 3:3], y], 1),
                torch.bfloat16,
                "npu",
            ),
        ]
        for fn, dtype, device in cases:
            with self.subTest(dtype=dtype, device=device, fn=fn):
                gm = trace(fn, dtype=dtype, device=device)
                before = gm.code
                rewrite(gm.graph)
                gm.recompile()
                self.assertEqual(gm.code, before)

    def test_missing_meta_and_mutation(self):
        for mutation in (False, True):
            gm = trace(pattern)
            if mutation:
                x = next(iter(gm.graph.nodes))
                output = next(n for n in gm.graph.nodes if n.op == "output")
                with gm.graph.inserting_before(output):
                    gm.graph.call_function(torch.ops.aten.add_.Tensor, (x, 1))
            else:
                for n in gm.graph.nodes:
                    if n.target == CAT:
                        n.meta.clear()
            gm.recompile()
            before = gm.code
            rewrite(gm.graph)
            gm.recompile()
            self.assertEqual(gm.code, before)

    def test_composition_and_registration_order(self):
        gm = self.check(pattern)
        multi_slice_concat_pass(gm.graph, max_segments=64, aclnn_copy_free_inputs=0)
        self.assertEqual(count(gm, MULTI_SLICE_CONCAT_TARGET), 1)
        self.assertEqual(count(gm, CAST), 1)
        names = [
            fn.__name__
            for levels in ASCEND_CUSTOME_PASS_REGISTER.values()
            for fns in levels.values()
            for fn in fns
        ]
        self.assertIn(rewrite.__name__, names)
        if multi_slice_concat_pass.__name__ in names:
            self.assertLess(
                names.index(rewrite.__name__),
                names.index(multi_slice_concat_pass.__name__),
            )

    def test_large_run_multiple_sources_and_chunking(self):
        def fn(x, z):
            parts = [
                torch.ops.aten.slice.Tensor(x if i % 2 else z, 1, i % 17, i % 17 + 1)
                for i in range(645)
            ]
            return torch.cat(parts + [x[:, :1].float()], 1)

        with FakeTensorMode():
            x = torch.empty(7, 19, dtype=torch.bfloat16, device="npu")
            gm = make_fx(fn)(x, x.clone())
        x, z = torch.randn(7, 19).bfloat16(), torch.randn(7, 19).bfloat16()
        expected = gm(x, z)
        rewrite(gm.graph)
        self.assertEqual(count(gm, CAST), 1)
        multi_slice_concat_pass(gm.graph, max_segments=64, aclnn_copy_free_inputs=0)
        gm.graph.lint()
        gm.recompile()
        self.assertEqual(count(gm, MULTI_SLICE_CONCAT_TARGET), 11)
        torch.testing.assert_close(gm(x, z), expected, rtol=0, atol=0)

    def test_pass_switch_changes_fx_cache_key(self):
        from torch_npu._inductor.fx_passes.graph_match_pass import AscendCustomPostPass

        enabled_key = AscendCustomPostPass().uuid()
        with spy_pass():
            self.assertNotEqual(enabled_key, AscendCustomPostPass().uuid())
        self.assertEqual(enabled_key, AscendCustomPostPass().uuid())

    def test_fold_cat_cannot_remove_cast_barrier(self):
        gm = trace(lambda x, y: pattern(x, y, nested=True))
        self.assertEqual(count(gm, CAT), 2)
        fold_cat(gm.graph)
        self.assertEqual(count(gm, CAT), 1)
        rewrite(gm.graph)
        self.assertEqual(count(gm, CAST), 1)
        gm.recompile()
        packed_code = gm.code
        fold_cat(gm.graph)
        gm.recompile()
        self.assertEqual(gm.code, packed_code)


@contextlib.contextmanager
def spy_pass():
    seen, originals = [], []

    def spy(graph):
        rewrite(graph)
        seen.extend(
            n for n in graph.nodes if n.target == CAST and n.args[0].target == CAT
        )

    spy.ignore_inference_check = True
    for levels in ASCEND_CUSTOME_PASS_REGISTER.values():
        for level, fns in levels.items():
            originals.append((levels, level, fns))
            levels[level] = [spy if fn is rewrite else fn for fn in fns]
    try:
        yield seen
    finally:
        for levels, level, fns in originals:
            levels[level] = fns


class TestCompile(unittest.TestCase):
    def test_dynamic_batch_and_backward(self):
        torch._dynamo.reset()
        with spy_pass() as seen:
            compiled = torch.compile(pattern, fullgraph=True, dynamic=True)
            for rows in (7, 17):
                x = torch.randn(
                    rows, 19, device="npu", dtype=torch.bfloat16, requires_grad=True
                )
                y = torch.randn(rows, 1, device="npu", requires_grad=True)
                torch._dynamo.mark_static(x, 1)
                expected = pattern(x, y)
                actual = compiled(x, y)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                grad = torch.ones_like(actual)
                ref_grad = torch.autograd.grad(expected, (x, y), grad)
                got_grad = torch.autograd.grad(actual, (x, y), grad)
                torch.testing.assert_close(got_grad, ref_grad, rtol=0, atol=0)
            self.assertTrue(seen, "compiled graph must actually hit the pass")


class TestGeneralizedGraph(unittest.TestCase):
    def verify(
        self, shape, cat_dim, source_dtype, output_dtype, explicit, slice_dim=None
    ):
        axis = cat_dim % len(shape)
        slice_axis = axis if slice_dim is None else slice_dim
        y_shape = list(shape)
        y_shape[axis] = 1
        if slice_axis != axis:
            y_shape[slice_axis] = 1

        def fn(x, y):
            parts = [
                torch.ops.aten.slice.Tensor(x, slice_axis, i, i + 1) for i in (3, 1, 3)
            ]
            if explicit:
                parts = [p.to(output_dtype) for p in parts]
            return torch.cat(parts + [y], cat_dim)

        with FakeTensorMode():
            gm = make_fx(fn)(
                torch.empty(shape, dtype=source_dtype, device="npu"),
                torch.empty(y_shape, dtype=output_dtype, device="npu"),
            )
        x = torch.randn(shape).to(source_dtype)
        y = torch.randn(y_shape).to(output_dtype)
        x.flatten()[:4] = torch.tensor(
            [float("nan"), float("inf"), -0.0, -float("inf")]
        )
        expected = gm(x, y)
        rewrite(gm.graph)
        gm.graph.lint()
        gm.recompile()
        self.assertEqual(count(gm, CAST), 1)
        actual = gm(x, y)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        self.assertTrue(torch.equal(torch.signbit(actual), torch.signbit(expected)))
        code = gm.code
        rewrite(gm.graph)
        gm.recompile()
        self.assertEqual(gm.code, code)

    def test_all_floating_conversion_pairs(self):
        dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        for src in dtypes:
            for dst in dtypes:
                if src == dst:
                    continue
                for explicit in (True, False):
                    if not explicit and torch.promote_types(src, dst) != dst:
                        continue
                    with self.subTest(src=src, dst=dst, explicit=explicit):
                        self.verify((5, 7), 1, src, dst, explicit)

    def test_rank_one_to_four_and_every_concat_axis(self):
        for shape in ((7,), (5, 7), (5, 6, 7), (5, 6, 7, 8)):
            for axis in range(len(shape)):
                for dim in (axis, axis - len(shape)):
                    with self.subTest(shape=shape, dim=dim):
                        self.verify(shape, dim, torch.float16, torch.float32, False)
        # A slice along a different axis is also a valid same-dtype concat input.
        self.verify((5, 6, 7), 2, torch.float32, torch.bfloat16, True, slice_dim=1)

    def test_source_dtype_changes_split_runs(self):
        def fn(a, b, c):
            parts = [
                a[:, :1].bfloat16(),
                a[:, 1:2].bfloat16(),
                b[:, :1].bfloat16(),
                b[:, 1:2].bfloat16(),
            ]
            return torch.cat(parts + [c], 1)

        with FakeTensorMode():
            gm = make_fx(fn)(
                torch.empty(5, 7, device="npu"),
                torch.empty(5, 7, device="npu", dtype=torch.float16),
                torch.empty(5, 1, device="npu", dtype=torch.bfloat16),
            )
        # Remove identity row slices as the post-grad pipeline does.
        for n in list(gm.graph.nodes):
            if n.target == torch.ops.aten.slice.Tensor and n.args[1] == 0:
                n.replace_all_uses_with(n.args[0])
                gm.graph.erase_node(n)
        gm.recompile()
        inputs = (
            torch.randn(5, 7),
            torch.randn(5, 7).half(),
            torch.randn(5, 1).bfloat16(),
        )
        expected = gm(*inputs)
        rewrite(gm.graph)
        gm.graph.lint()
        gm.recompile()
        self.assertEqual(count(gm, CAST), 2)
        torch.testing.assert_close(gm(*inputs), expected, rtol=0, atol=0)

    def test_nonfloating_and_device_copy_are_left_unchanged(self):
        for dtype in (torch.int32, torch.int64, torch.bool, torch.complex64):
            gm = trace(pattern, dtype=dtype)
            before = gm.code
            rewrite(gm.graph)
            gm.recompile()
            self.assertEqual(before, gm.code)
        gm = trace(lambda x, y: pattern(x, y, explicit=True))
        for n in gm.graph.nodes:
            if n.target == torch.ops.aten._to_copy.default:
                n.kwargs = dict(n.kwargs, non_blocking=True)
        gm.recompile()
        before = gm.code
        rewrite(gm.graph)
        gm.recompile()
        self.assertEqual(before, gm.code)


class TestGeneralizedCompile(unittest.TestCase):
    def test_rank_three_fp16_promotion_and_fp32_narrowing(self):
        for source_dtype, output_dtype in (
            (torch.float16, torch.float32),
            (torch.float32, torch.bfloat16),
        ):

            def fn(x, y):
                parts = [x.narrow(1, i, 1).to(output_dtype) for i in (3, 1, 3)]
                return torch.cat(parts + [y], 1)

            torch._dynamo.reset()
            with spy_pass() as seen:
                compiled = torch.compile(fn, fullgraph=True, dynamic=True)
                for rows in (5, 9):
                    x = torch.randn(
                        rows, 7, 3, device="npu", dtype=source_dtype, requires_grad=True
                    )
                    y = torch.randn(
                        rows, 1, 3, device="npu", dtype=output_dtype, requires_grad=True
                    )
                    torch._dynamo.mark_static(x, 1)
                    expected, actual = fn(x, y), compiled(x, y)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    grad = torch.ones_like(actual)
                    reference = torch.autograd.grad(expected, (x, y), grad)
                    result = torch.autograd.grad(actual, (x, y), grad)
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)
                self.assertTrue(seen, "generalized compiled graph must hit the pass")


if __name__ == "__main__":
    unittest.main()
