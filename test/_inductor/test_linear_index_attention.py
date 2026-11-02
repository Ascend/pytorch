"""Linear Triton regressions for attention subgraphs containing index."""

import contextlib
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sympy

import torch
import torch_npu
import torch_npu._inductor
from torch._inductor import config
from torch._inductor.utils import run_and_get_code


def eager_query_key(args):
    bias, query, index, cross = args
    query = torch.ops.aten.add.Tensor(bias, query)
    query = query.reshape(115, 1, 6, 128).permute(0, 2, 1, 3)
    query = query.reshape(690, 1, 128).float().unsqueeze(-1)
    cross = cross.reshape(1, 202, 6, 128).permute(0, 2, 1, 3)
    index = index.unsqueeze(1).expand(23, 5)
    index = index.clone(memory_format=torch.contiguous_format).reshape(115)
    cross = torch.ops.aten.index.Tensor(cross, [index])
    cross = cross.clone(memory_format=torch.contiguous_format)
    cross = cross.permute(0, 1, 3, 2).reshape(690, 128, 202)
    cross = cross.float().unsqueeze(1)
    return ((query * cross).sum(dim=2),)


def eager_indexed_value_cast(args):
    weights, index, cross = args
    weights = weights.unsqueeze(-1)
    cross = cross.reshape(1, 202, 6, 128).permute(0, 2, 1, 3)
    index = index.unsqueeze(1).expand(23, 5)
    index = index.clone(memory_format=torch.contiguous_format).reshape(115)
    cross = torch.ops.aten.index.Tensor(cross, [index])
    cross = cross.clone(memory_format=torch.contiguous_format)
    cross = cross.reshape(690, 202, 128).float().unsqueeze(1)
    return ((weights * cross).sum(dim=2).half(),)


def make_inputs(case, seed):
    # Build backing storage on CPU, then transfer it before as_strided so the
    # exact network input strides, including size-one strides, are preserved.
    generator = torch.Generator().manual_seed(seed)

    def tensor(shape, stride, dtype=torch.float32):
        size = 1 + sum((dim - 1) * step for dim, step in zip(shape, stride))
        raw = torch.randn(size, generator=generator, dtype=torch.float32) * 0.2
        return raw.to(dtype=dtype, device="npu").as_strided(shape, stride)

    cross = tensor((202, 768), (768, 1), torch.float16)
    # Both 0 and -1 legally index the batch axis of size one; vary within a beam.
    index = -torch.randint(0, 2, (23,), generator=generator).to("npu")
    if case == "query_key":
        return [
            tensor((768,), (1,), torch.float16),
            tensor((115, 768), (768, 1), torch.float16),
            index,
            cross,
        ]
    if case == "indexed_value_cast":
        return [tensor((690, 1, 202), (1, 139380, 690)), index, cross]
    raise ValueError(case)


@contextlib.contextmanager
def observe_linear_codegen(failures):
    from torch_npu._inductor.codegen.scheduling import NPUTritonScheduling

    original = NPUTritonScheduling.codegen_node

    def observe(self, node):
        try:
            return original(self, node)
        except Exception as exc:
            failures.append((node.get_name(), type(exc).__name__, str(exc)))
            raise

    NPUTritonScheduling.codegen_node = observe
    try:
        yield
    finally:
        NPUTritonScheduling.codegen_node = original


class TestSortedAxisOrder(unittest.TestCase):
    def make_kernel(self, *, mapping=None, strides=(155136, 128, 768),
                    order=("y0", "x4", "x3", "r2")):
        from torch._inductor.ir import FloorDiv
        from torch._inductor.loop_body import MemoryUsageType

        r2, x4, y0, x3, tmp8 = sympy.symbols("r2 x4 y0 x3 tmp8", integer=True)
        symbols = [r2, x4, y0, x3]
        if "r1" in order:
            symbols.append(sympy.Symbol("r1", integer=True))
        axes = {
            symbol: SimpleNamespace(
                name=str(symbol), prefix=str(symbol)[0], expr=symbol,
                symbol=lambda symbol=symbol: symbol,
            )
            for symbol in symbols
        }
        indirect_stride, x4_stride, y0_stride = strides
        body = SimpleNamespace(
            indexing={"load": indirect_stride * tmp8 + x4_stride * x4
                      + y0_stride * y0 + sum(1000000 * s for s in symbols if str(s).startswith("r"))},
            indirect_replacements={
                tmp8: FloorDiv(x4 + 6 * x3, 30) if mapping is None else mapping(x3, x4)
            },
            memory_usage={MemoryUsageType.LOAD: [SimpleNamespace(index_name="load")]},
        )
        by_name = {axis.name: axis for axis in axes.values()}
        return SimpleNamespace(
            range_tree_nodes=axes, sorted_axis=[by_name[name] for name in order],
            node_schedule=[SimpleNamespace(_body=body)],
        )

    def assert_order(self, kernel, expected):
        from torch_npu._inductor.codegen.split_tiling import reorder_sorted_axis

        original = kernel.range_tree_nodes
        original_items = list(original.items())
        original_sorted = list(kernel.sorted_axis)
        self.assertTrue(reorder_sorted_axis(kernel))
        self.assertEqual([axis.name for axis in kernel.sorted_axis], expected.split())
        self.assertIs(kernel.range_tree_nodes, original)
        self.assertEqual(list(original.items()), original_items)
        self.assertEqual({id(axis) for axis in kernel.sorted_axis}, {id(axis) for axis in original.values()})
        for i, axis in enumerate(original_sorted):
            if axis.prefix == "r":
                self.assertIs(kernel.sorted_axis[i], axis)

    def test_combined_stride_weights(self):
        from torch._inductor.ir import FloorDiv

        cases = [
            ({}, "x3 x4 y0 r2"),
            ({"mapping": lambda x3, x4: x3 + x4 / 10000000}, "x3 y0 x4 r2"),
            ({"mapping": lambda x3, x4: x3 + FloorDiv(x4, 10000000)}, "x3 y0 x4 r2"),
            ({"mapping": lambda x3, x4: x3 + x4 / 10000000,
              "strides": (10, 128, 768)}, "y0 x4 x3 r2"),
            ({"mapping": lambda x3, x4: x3 + x4 / 2,
              "strides": (100, 60, 80)}, "x4 x3 y0 r2"),
            ({"mapping": lambda x3, x4: x3 + x4,
              "strides": (100, -80, 50)}, "x3 y0 x4 r2"),
        ]
        for kwargs, expected in cases:
            with self.subTest(expected=expected, strides=kwargs.get("strides")):
                self.assert_order(self.make_kernel(**kwargs), expected)

    def test_reduction_slots_are_preserved(self):
        for order, expected in [
            (("r2", "y0", "x4", "x3"), "r2 x3 x4 y0"),
            (("x4", "r2", "y0", "x3"), "x3 r2 x4 y0"),
            (("y0", "r2", "x4", "r1", "x3"), "x3 r2 x4 r1 y0"),
        ]:
            with self.subTest(order=order):
                self.assert_order(self.make_kernel(order=order), expected)

    def test_incomplete_axes_are_rejected(self):
        from torch._inductor.ir import FloorDiv
        from torch_npu._inductor.codegen.split_tiling import reorder_sorted_axis

        for missing in ("reduction", "output", "sorted_axis"):
            with self.subTest(missing=missing):
                kernel = self.make_kernel()
                body = kernel.node_schedule[0]._body
                if missing == "reduction":
                    r2 = next(s for s in kernel.range_tree_nodes if str(s) == "r2")
                    body.indexing["load"] -= 1000000 * r2
                elif missing == "output":
                    tmp8 = next(iter(body.indirect_replacements))
                    x4 = next(s for s in kernel.range_tree_nodes if str(s) == "x4")
                    body.indirect_replacements[tmp8] = FloorDiv(x4, 30)
                else:
                    kernel.sorted_axis.pop(0)
                original = kernel.sorted_axis
                self.assertFalse(reorder_sorted_axis(kernel))
                self.assertIs(kernel.sorted_axis, original)

    def test_tiling_initializes_sorted_order(self):
        from torch_npu._inductor.codegen.split_tiling import SplitTiling

        for complete in (True, False):
            with self.subTest(complete=complete):
                kernel = self.make_kernel()
                if not complete:
                    kernel.node_schedule[0]._body.indirect_replacements.clear()
                original = list(kernel.range_tree_nodes.items())
                with (
                    patch.object(SplitTiling, "find_lowest_dimension"),
                    patch.object(SplitTiling, "is_contiguous_reduction", return_value=False),
                ):
                    SplitTiling(kernel)
                expected = "x3 x4 y0 r2" if complete else "y0 x4 x3 r2"
                self.assertEqual([axis.name for axis in kernel.sorted_axis], expected.split())
                self.assertEqual([axis.sorted_order for axis in kernel.sorted_axis], [0, 1, 2, 3])
                self.assertEqual(list(kernel.range_tree_nodes.items()), original)


@unittest.skipUnless(torch.npu.is_available(), "requires an Ascend NPU")
class TestLinearIndexAttention(unittest.TestCase):
    def setUp(self):
        from torch_npu._inductor import config as npu_config

        if not npu_config.inductor_indirect_memory_mode:
            self.skipTest("requires enabled NPU indirect-memory lowering")
        torch_npu._inductor._load_triton_backend()
        torch.npu.set_device(0)
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def check_eager_compile(self, case, eager, *, autotune=True, output_stride=None):
        failures = []
        options = {
            "fx_graph_cache": False,
            "cpp_wrapper": False,
            "max_autotune": autotune,
            "coordinate_descent_tuning": autotune,
        }
        if not autotune:
            options["triton.persistent_reductions"] = False
        # Equality alone cannot detect linear failures hidden by no_linear.
        with (
            config.patch(options),
            observe_linear_codegen(failures),
            torch.no_grad(),
        ):
            compiled = torch.compile(
                eager, backend="inductor", fullgraph=True, dynamic=False
            )
            def call_args(seed):
                args = [make_inputs(case, seed)]
                if output_stride is not None:
                    args.append(torch.empty_strided(
                        (690, 1, 202), output_stride, dtype=torch.float32, device="npu"
                    ))
                return args

            _, codes = run_and_get_code(compiled, *call_args(0))
            torch.npu.synchronize()
            for seed in range(3):
                actual, expected = compiled(*call_args(seed)), eager(*call_args(seed))
                torch.npu.synchronize()
                self.assertEqual(len(actual), len(expected))
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(
                        result,
                        reference,
                        atol=1e-3,
                        rtol=1e-3,
                        check_stride=True,
                    )
        self.assertFalse(failures, str(failures))
        modes = re.findall(
            r"['\"]inductor_ascend_linear_mode['\"]\s*:\s*['\"]([^'\"]+)",
            "\n".join(codes),
        )
        self.assertIn("linear", modes, "no linear DSL was emitted")
        self.assertNotIn("no_linear", modes)

    def test_gold_axes_after_expand_and_index(self):
        self.check_eager_compile("query_key", eager_query_key)

    def test_query_key_with_historical_output_stride(self):
        def eager_with_output(args, output):
            output.copy_(eager_query_key(args)[0])
            return (output,)

        self.check_eager_compile(
            "query_key", eager_with_output, output_stride=(1, 139380, 690)
        )

    def test_gold_axes_with_sum_cast_epilogue(self):
        self.check_eager_compile("indexed_value_cast", eager_indexed_value_cast)

    def test_indexed_sum_cast_after_no_loop_channel(self):
        self.check_eager_compile(
            "indexed_value_cast", eager_indexed_value_cast, autotune=False
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
