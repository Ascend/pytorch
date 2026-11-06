"""Regression coverage for index -> reshape -> index graph rewriting."""

import unittest
from contextlib import contextmanager
from math import prod
from unittest.mock import patch

import torch
import torch_npu
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.utils import counters
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_leaves

from torch_npu._inductor import config
from torch_npu._inductor.fx_passes.ascend_custom_passes.index_reshape_index import (
    TARGET,
    _is_index_reshape_index_profitable,
    index_reshape_index_pass,
    index_reshape_index_reference,
)
from torch_npu._inductor.fx_passes.graph_match_pass import AscendCustomPostPass


DTYPES = (torch.float16, torch.bfloat16, torch.float32)
INDEX_DTYPES = (torch.int32, torch.int64)
requires_npu = unittest.skipUnless(torch_npu.npu.is_available(), "requires NPU")
requires_a5 = unittest.skipUnless(config.is_ascend950, "requires Ascend950")


def subgraph(input_tensor, first_indices, second_indices):
    selected = torch.ops.aten.index.Tensor(input_tensor, [first_indices])
    flattened = torch.ops.aten.reshape.default(
        selected, [first_indices.numel() * input_tensor.shape[1], input_tensor.shape[2]]
    )
    return torch.ops.aten.index.Tensor(flattened, [second_indices])


def flatten_subgraph(input_tensor, first_indices, second_indices):
    selected = input_tensor[first_indices]
    return selected.reshape(-1, input_tensor.shape[-1])[second_indices]


def wrong_flatten(input_tensor, first_indices, second_indices):
    return input_tensor[first_indices].reshape(-1, input_tensor.shape[-1] * 2)[
        second_indices // 2
    ]


def shared_intermediate(input_tensor, first_indices, second_indices):
    selected = input_tensor[first_indices]
    return selected.reshape(-1, input_tensor.shape[-1])[second_indices], selected


class TestIndexReshapeIndex(unittest.TestCase):
    @staticmethod
    def inputs(
        dtype=torch.bfloat16, index_dtype=torch.int64, second_count=65536,
        *, shape=(2, 1024, 64), first_count=128, dyadic=False,
    ):
        if dyadic:
            input_tensor = torch.randint(-2, 3, shape, device="npu").to(dtype) / 8
        else:
            input_tensor = torch.randn(shape, device="npu", dtype=dtype)
        first = torch.randint(
            -shape[0], shape[0], (first_count,), device="npu", dtype=index_dtype
        )
        count = first_count * prod(shape[1:-1])
        second = torch.randint(
            -count, count, (second_count,), device="npu", dtype=index_dtype
        )
        return input_tensor, first, second

    def strided_inputs(self, dtype):
        source, first, second = self.inputs(dtype)
        return (
            torch.cat((source, source))[2:],
            first.repeat_interleave(2)[::2], second.repeat_interleave(2)[::2],
        )

    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for a, b in zip(tree_leaves(actual), tree_leaves(expected)):
            if isinstance(a, torch.Tensor):
                self.assertEqual(a.stride(), b.stride())

    def rewrite(self, gm):
        index_reshape_index_pass(gm.graph)
        gm.graph.lint()
        gm.recompile()
        return sum(n.target == TARGET for n in gm.graph.nodes)

    def apply(self, fn, args, tracing_mode="fake"):
        gm = make_fx(fn, tracing_mode=tracing_mode)(*args)
        return gm, self.rewrite(gm)

    @contextmanager
    def compile_session(self, training=False, hits=None):
        torch._dynamo.reset()
        before = counters["inductor"]["npu_index_reshape_index"]
        try:
            with torch.set_grad_enabled(training):
                yield
            self.assertEqual(
                counters["inductor"]["npu_index_reshape_index"] - before,
                int(config.is_ascend950) if hits is None else hits,
            )
        finally:
            torch._dynamo.reset()

    @staticmethod
    def compile(fn, **options):
        calls = []

        def backend(gm, example_inputs):
            calls.append(1)
            return lookup_backend("inductor")(gm, example_inputs)

        return torch.compile(fn, backend=backend, fullgraph=True, **options), calls

    def assert_gradients(self, actual, expected, args, repeats=1, scale=1):
        trainable = [t for t in args if t.requires_grad]
        grads = []
        for outputs in (actual, expected):
            outputs = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
            for t in trainable:
                t.grad = None
            upstream = tuple(torch.ones_like(t) / scale for t in outputs)
            for step in range(repeats):
                torch.autograd.backward(
                    outputs, upstream, retain_graph=step < repeats - 1
                )
            grads.append(tuple(t.grad.clone() for t in trainable))
        torch.testing.assert_close(*grads, rtol=0, atol=0)

    def assert_compiled(self, fn, args, training=False, hits=None, **options):
        with self.compile_session(training, hits):
            compiled, calls = self.compile(fn, **options)
            actual, expected = compiled(*args), fn(*args)
            self.assert_exact(actual, expected)
            if training:
                self.assert_gradients(actual, expected, args)
            self.assertEqual(len(calls), 1)

    def test_generalized_reference(self):
        for shape in ((2, 8), (2, 3, 8), (2, 3, 4, 8), (2, 3, 2, 4, 8)):
            with self.subTest(shape=shape):
                input_tensor = torch.arange(prod(shape)).reshape(shape)
                first_indices = torch.tensor([-1, 0, -2, -1])
                block_size = prod(shape[1:-1])
                count = first_indices.numel() * block_size
                second_indices = torch.tensor([-count, -1, 0, count - 1, 0])
                expected = input_tensor[first_indices].flatten(0, -2)[second_indices]
                actual = index_reshape_index_reference(
                    input_tensor, first_indices, second_indices
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # Independently check the flattened-index composition used by lowering.
                normalized = second_indices % count
                combined = (
                    (first_indices[normalized // block_size] % shape[0]) * block_size
                    + normalized % block_size
                )
                torch.testing.assert_close(
                    input_tensor.flatten(0, -2)[combined], expected, rtol=0, atol=0
                )

    @requires_npu
    def test_generalized_input_rank(self):
        for shape, first_count, second_count in (
            ((2, 8192), 512, 512), ((2, 16, 64, 64), 128, 65536),
            ((2, 4, 4, 64, 64), 128, 65536),
        ):
            with self.subTest(shape=shape):
                args = self.inputs(shape=shape, first_count=first_count,
                                   second_count=second_count)
                gm, matches = self.apply(flatten_subgraph, args)
                self.assertEqual(matches, int(config.is_ascend950))
                self.assert_exact(gm(*args), flatten_subgraph(*args))
                self.assertEqual(self.apply(wrong_flatten, args)[1], 0)
                # A 2D flatten is removed before the post pass; compile its
                # rewritten FX graph to still exercise the internal lowering.
                fn = gm if len(shape) == 2 else flatten_subgraph
                self.assert_compiled(fn, args,
                                     hits=int(config.is_ascend950 and len(shape) > 2))

    def test_byte_profitability_cases(self):
        # Meta tensors exercise byte-volume thresholds without allocating data.
        # Small feature dimensions can qualify when the first index result is large.
        cases = (
            ("small_features_large_intermediate", (2, 1000, 32), 378, 131072,
             torch.bfloat16, torch.int64, True),
            ("fp32_small_features", (2, 1000, 16), 378, 131072,
             torch.float32, torch.int64, True),
            ("bf16_small_output", (2, 1000, 16), 378, 131072,
             torch.bfloat16, torch.int64, False),
            ("small_output", (2, 1000, 256), 378, 4096,
             torch.bfloat16, torch.int64, False),
            ("no_size_increase", (2, 1000, 32), 2, 131072,
             torch.bfloat16, torch.int64, False),
            ("index_work_dominates", (2, 1000, 32), 378, 1048576,
             torch.bfloat16, torch.int64, False),
            ("small_intermediate", (2, 16, 64), 128, 65536,
             torch.bfloat16, torch.int64, False),
            ("int32_index_budget", (2, 1000, 32), 170, 131072,
             torch.bfloat16, torch.int32, True),
            ("int64_index_budget", (2, 1000, 32), 170, 131072,
             torch.bfloat16, torch.int64, False),
        )
        for (
            name, shape, first_count, second_count, dtype, index_dtype, expected
        ) in cases:
            with self.subTest(name=name):
                input_tensor = torch.empty(shape, device="meta", dtype=dtype)
                first_indices = torch.empty(
                    first_count, device="meta", dtype=index_dtype
                )
                second_indices = torch.empty(
                    second_count, device="meta", dtype=index_dtype
                )
                self.assertEqual(
                    _is_index_reshape_index_profitable(
                        input_tensor, first_indices, second_indices
                    ),
                    expected,
                )

    def test_cache_fallback_mode(self):
        with patch.object(config, "enable_full_lowering_fallback", ""):
            normal = AscendCustomPostPass().uuid()
        with patch.object(config, "enable_full_lowering_fallback", "allfallback"):
            fallback = AscendCustomPostPass().uuid()
        self.assertNotEqual(normal, fallback)

    def test_reference_negative_and_invalid_indices(self):
        input_tensor = torch.arange(24).reshape(2, 3, 4)
        first_indices = torch.tensor([-1, 0, -2])
        second_indices = torch.tensor([-9, -1, 0, 8])
        torch.testing.assert_close(
            index_reshape_index_reference(input_tensor, first_indices, second_indices),
            subgraph(input_tensor, first_indices, second_indices),
        )
        for invalid in (
            torch.tensor([-10]),
            torch.tensor([9]),
            torch.tensor([2**32]),
            torch.tensor([-(2**32)]),
        ):
            with self.assertRaises(IndexError):
                index_reshape_index_reference(input_tensor, first_indices, invalid)
        with self.assertRaises(IndexError):
            index_reshape_index_reference(
                input_tensor, torch.tensor([0, 2]), second_indices[:1]
            )

    @requires_npu
    def test_architecture_gate(self):
        args = self.inputs()
        # Exercise both matcher branches on every NPU generation. The positive
        # control prevents non-A5 runs from passing only because nothing matches.
        for is_ascend950 in (False, True):
            with (
                self.subTest(is_ascend950=is_ascend950),
                patch.object(config, "is_ascend950", is_ascend950),
            ):
                gm, count = self.apply(subgraph, args)
                self.assertEqual(count, int(is_ascend950))
                torch.testing.assert_close(gm(*args), subgraph(*args), rtol=0, atol=0)

    @requires_npu
    def test_match_and_exact_output(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for index_dtype in (torch.int32, torch.int64):
                with self.subTest(dtype=dtype, index_dtype=index_dtype):
                    args = self.inputs(dtype, index_dtype)
                    gm, count = self.apply(subgraph, args)
                    # The optimization is A5-only; other NPUs retain ATen nodes.
                    self.assertEqual(count, int(config.is_ascend950))
                    torch.testing.assert_close(
                        gm(*args), subgraph(*args), rtol=0, atol=0
                    )

    @requires_npu
    def test_no_match_cases(self):
        args = self.inputs()
        self.assertEqual(self.apply(subgraph, self.inputs(second_count=7))[1], 0)
        self.assertEqual(self.apply(subgraph, self.inputs(second_count=4096))[1], 0)
        self.assertEqual(self.apply(subgraph, self.inputs(second_count=0))[1], 0)
        no_size_increase = (args[0], args[1][:2], args[2] % 2048)
        self.assertEqual(self.apply(subgraph, no_size_increase)[1], 0)
        small_feature_input = torch.randn(
            2, 1024, 32, device="npu", dtype=torch.bfloat16
        )
        narrow_args = (small_feature_input, args[1], args[2].repeat(2))
        self.assertEqual(self.apply(subgraph, narrow_args)[1], 0)
        noncontiguous = args[0].transpose(1, 2)
        self.assertEqual(self.apply(subgraph, (noncontiguous, args[1], args[2]))[1], 0)
        cpu_args = tuple(t.cpu() for t in args)
        self.assertEqual(self.apply(subgraph, cpu_args)[1], 0)
        self.assertEqual(self.apply(subgraph, self.inputs(torch.float64))[1], 0)
        integer_input = args[0].to(torch.int32)
        self.assertEqual(self.apply(subgraph, (integer_input, *args[1:]))[1], 0)
        matrix_first_indices = args[1].reshape(16, 8)
        self.assertEqual(
            self.apply(subgraph, (args[0], matrix_first_indices, args[2]))[1], 0
        )
        matrix_second_indices = args[2].reshape(256, 256)
        self.assertEqual(self.apply(subgraph, (*args[:2], matrix_second_indices))[1], 0)
        grad_input = args[0].detach().requires_grad_(True)
        gm = make_fx(subgraph, tracing_mode="fake")(grad_input, *args[1:])
        # Training metadata must no longer prevent this post-AOT rewrite.
        placeholder = next(n for n in gm.graph.nodes if n.op == "placeholder")
        placeholder.meta["val"].requires_grad_(True)
        index_reshape_index_pass(gm.graph)
        self.assertEqual(
            sum(n.target == TARGET for n in gm.graph.nodes), int(config.is_ascend950)
        )

        self.assertEqual(self.apply(wrong_flatten, args)[1], 0)
        self.assertEqual(self.apply(shared_intermediate, args)[1], 0)
        with patch.object(config, "is_ascend950", False):
            self.assertEqual(self.apply(subgraph, args)[1], 0)
        with patch.object(config, "enable_full_lowering_fallback", "allfallback"):
            self.assertEqual(self.apply(subgraph, args)[1], 0)

    @requires_npu
    def test_small_features_large_intermediate(self):
        args = self.inputs(shape=(2, 1000, 32), first_count=378, second_count=131072)
        gm, count = self.apply(subgraph, args)
        self.assertEqual(count, int(config.is_ascend950))
        self.assert_exact(gm(*args), subgraph(*args))
        self.assert_compiled(subgraph, args)

    @requires_npu
    def test_training_values_and_gradients(self):
        for dtype in DTYPES:
            for index_dtype in INDEX_DTYPES:
                with self.subTest(dtype=dtype, index_dtype=index_dtype):
                    source, first, second = self.inputs(dtype, index_dtype)
                    # Duplicate/negative indices and exactly representable gradients.
                    args = (source.requires_grad_(), first, second[:1024].repeat(64))
                    self.assert_compiled(subgraph, args, training=True)

    @requires_npu
    def test_training_dynamic_saved_outputs(self):
        def fn(input_tensor, addend, first_indices, second_indices):
            result = flatten_subgraph(
                input_tensor + addend, first_indices, second_indices
            )
            # square saves the rewritten result; the source has another grad branch.
            return result, result.square(), input_tensor.sum() * 0.125

        for cases in (
            (((2, 1024, 64), 128, 65536), ((3, 1000, 80), 160, 69632)),
            (((2, 8, 127, 64), 128, 65536), ((3, 10, 101, 80), 160, 69632)),
        ):
            with (
                self.subTest(rank=len(cases[0][0])), self.compile_session(training=True)
            ):
                compiled, calls = self.compile(fn, dynamic=True)
                for shape, first_count, second_count in cases:
                    source, first, second = self.inputs(
                        torch.float32, shape=shape, first_count=first_count,
                        second_count=second_count, dyadic=True,
                    )
                    addend = torch.randint(-2, 3, (1, *shape[1:]), device="npu")
                    addend = (addend.float() / 8).requires_grad_()
                    args = (source.requires_grad_(), addend, first, second)
                    actual, expected = compiled(*args), fn(*args)
                    self.assert_exact(actual, expected)
                    # Dyadic values keep repeated backward/accumulation exact.
                    self.assert_gradients(actual, expected, args, repeats=2, scale=8)
                self.assertEqual(len(calls), 1)

    @requires_npu
    def test_intermediate_size_users(self):
        args = self.inputs()
        gm = make_fx(subgraph, tracing_mode="fake")(*args)
        output = next(n for n in gm.graph.nodes if n.op == "output")
        root = output.args[0]
        view = root.args[0]
        first_index = view.args[0]
        queries = []
        with gm.graph.inserting_before(root):
            for node, dim in (
                (first_index, 0), (first_index, -1), (view, 0), (view, -1)
            ):
                query = gm.graph.call_function(torch.ops.aten.sym_size.int, (node, dim))
                query.meta["val"] = node.meta["val"].shape[dim]
                queries.append(query)
        output.args = ((root, *queries),)
        gm.recompile()
        expected = gm(*args)
        matches = self.rewrite(gm)
        torch.testing.assert_close(gm(*args), expected, rtol=0, atol=0)
        self.assertEqual(
            matches, int(config.is_ascend950)
        )

    @requires_npu
    def test_training_shared_intermediate_not_rewritten(self):
        source, first, second = self.inputs(torch.float32)
        self.assert_compiled(
            shared_intermediate, (source.requires_grad_(), first, second),
            training=True, hits=0,
        )

    @requires_npu
    def test_duplicate_indices_and_shared_result(self):
        def fn(input_tensor, addend, first_indices, second_indices):
            result = subgraph(input_tensor + addend, first_indices, second_indices)
            return result, result * 2

        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                source, first, second = self.inputs(dtype)
                addend = torch.randn(1, 1024, 64, device="npu", dtype=dtype)
                self.assert_compiled(
                    fn, (source, addend, first, second[:128].repeat(512))
                )

    @requires_npu
    @unittest.skipIf(config.is_ascend950, "requires non-A5 fallback path")
    def test_non_a5_compiled_fallback(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                self.assert_compiled(subgraph, self.strided_inputs(dtype), hits=0)

    @requires_npu
    @requires_a5
    def test_compiled_values_strides_and_epilogue(self):
        def fn(input_tensor, addend, first_indices, second_indices):
            return subgraph(input_tensor + addend, first_indices, second_indices) + 1

        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                source, first, second = self.strided_inputs(dtype)
                addend = torch.randn(1, 1024, 64, device="npu", dtype=dtype)
                self.assert_compiled(fn, (source, addend, first, second))

    @requires_npu
    @requires_a5
    def test_dynamic_match_and_reuse(self):
        for cases in (
            (((2, 1000, 64), 378, 131072), ((3, 1024, 80), 400, 139264)),
            (((2, 8, 125, 64), 378, 131072), ((3, 16, 64, 80), 400, 139264)),
        ):
            with self.subTest(rank=len(cases[0][0])):
                self._check_dynamic_match_and_reuse(cases)

    def _check_dynamic_match_and_reuse(self, cases):
        with self.compile_session():
            compiled, calls = self.compile(flatten_subgraph, dynamic=True)
            for shape, first_count, second_count in cases:
                args = self.inputs(shape=shape, first_count=first_count,
                                   second_count=second_count)
                self.assert_exact(compiled(*args), flatten_subgraph(*args))
            self.assertEqual(len(calls), 1)
            # Cross the profitability guard, then reuse the original fused graph.
            small_args = (args[0], args[1][:2],
                          args[2][:4096] % (2 * prod(shape[1:-1])))
            self.assert_exact(compiled(*small_args), flatten_subgraph(*small_args))
            self.assert_exact(compiled(*args), flatten_subgraph(*args))
            self.assertEqual(len(calls), 2)

    @requires_npu
    def test_symbolic_reshape_equivalence(self):
        args = self.inputs()
        for fn, expected in (
            (flatten_subgraph, int(config.is_ascend950)), (wrong_flatten, 0)
        ):
            with self.subTest(fn=fn.__name__):
                self.assertEqual(
                    self.apply(fn, args, tracing_mode="symbolic")[1], expected
                )

    @requires_npu
    def test_dynamic_falls_back_for_two_shapes(self):
        with self.compile_session(hits=0):
            compiled, calls = self.compile(subgraph, dynamic=True)
            for count in (4096, 4352):
                args = self.inputs(second_count=count)
                self.assert_exact(compiled(*args), subgraph(*args))
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
