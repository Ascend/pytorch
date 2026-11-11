import unittest
from unittest import mock

import torch
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize

from torch_npu._inductor.fx_passes.ascend_custom_passes import ascend_graph_pass as passes


def index_table(offsets, widths, device="cpu"):
    """The column table the pass would register for this plan."""
    return torch.tensor(passes.index_plan_columns(offsets, widths),
                        dtype=torch.int32, device=device)


@instantiate_parametrized_tests
class TestIndexedSliceConcatPlans(unittest.TestCase):
    @parametrize("columns", [[8], [-9], [0, 8], [3, 100]])
    def test_column_outside_source_rejected(self, columns):
        # Direct eager calls use aten's backend-specific bounds error; the pass
        # validates its column plan before creating this internal operator.
        with self.assertRaises((RuntimeError, IndexError)):
            table = torch.tensor(columns, dtype=torch.int32)
            passes.INDEXED_SLICE_CONCAT_TARGET(
                torch.empty(3, 8), table, 0, table.numel(), torch.float32
            )

    @parametrize(
        "plan",
        [([], 0, 0), ([0], 0, 0), ([0], 0, -1),
         ([0], -1, 1), ([0], 1, 1), ([0], 0, 2)],
    )
    def test_invalid_window_skipped(self, plan):
        columns, offset, count = plan
        table = torch.tensor(columns, dtype=torch.int32)
        with self.assertLogs(passes.log.name, level="DEBUG"):
            self.assertFalse(passes._msc_valid_index_table(table, offset, count))

    @parametrize("dtype", [torch.int32, torch.int64])
    def test_only_selected_window_is_checked(self, dtype):
        src = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        table = torch.tensor([-1, 0, 7, 8], dtype=dtype)
        actual = passes.INDEXED_SLICE_CONCAT_TARGET(src, table, 1, 2, src.dtype)
        torch.testing.assert_close(actual, src[:, [0, 7]], rtol=0, atol=0)

    @parametrize("mode", ["fake", "functional_fake", "meta"])
    def test_metadata_propagation_does_not_read_table(self, mode):
        if mode != "meta":
            fake_mode = FakeTensorMode()
            src = fake_mode.from_tensor(torch.empty(3, 8))
            table = fake_mode.from_tensor(torch.tensor([-1, 8], dtype=torch.int32))
            if mode == "functional_fake":
                src = torch._to_functional_tensor(src)
                table = torch._to_functional_tensor(table)
        else:
            src = torch.empty(3, 8, device="meta")
            table = torch.empty(2, dtype=torch.int32, device="meta")
        with mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("table data read")):
            # Exercise the reference directly too: dispatch may unwrap functional
            # tensors before calling a CompositeExplicitAutograd implementation.
            actual = passes._indexed_slice_concat_ref(src, table, 0, 2, torch.float16)
            through_dispatch = passes.INDEXED_SLICE_CONCAT_TARGET(src, table, 0, 2, torch.float16)
        self.assertEqual(actual.shape, (3, 2))
        self.assertEqual(actual.dtype, torch.float16)
        self.assertEqual(through_dispatch.shape, actual.shape)
        self.assertTrue(actual.is_meta if mode == "meta" else is_fake(actual))
        with self.assertLogs(passes.log.name, level="DEBUG"):
            self.assertFalse(passes._msc_valid_index_table(table, 1, 2))

    @parametrize("columns", [1048576, 1048577])
    def test_index_table_limit(self, columns):
        def fn(x):
            return torch.cat([x[:, :columns - 1], x[:, -1:]], 1)

        graph = make_fx(fn, tracing_mode="fake")(torch.empty(1, columns)).graph
        cat = next(n for n in graph.nodes if n.target is torch.ops.aten.cat.default)
        runs = passes._msc_index_runs(cat.args[0], torch.float32, 1)
        self.assertEqual(len(runs), int(columns == 1048576))

    @parametrize("shape", [(8,), (2, 3, 8)])
    def test_non_matrix_skipped(self, shape):
        with self.assertLogs(passes.log.name, level="DEBUG"):
            self.assertFalse(passes._msc_valid_index_columns(torch.empty(shape), [0]))


@instantiate_parametrized_tests
@unittest.skipUnless(hasattr(torch, "npu") and torch.npu.is_available(), "requires NPU")
class TestIndexedSliceConcatBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from torch_npu.utils._dynamo import _InductorNpuRegistry
        from torch_npu._inductor.kernel.multi_slice_concat import (
            _register_npu_inductor_multi_slice_concat,
        )
        _InductorNpuRegistry.register_inductor_npu()
        _register_npu_inductor_multi_slice_concat()

    def setUp(self):
        torch._dynamo.reset()

    def assert_exact(self, actual, expected):
        actual, expected = actual.cpu(), expected.cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        # assert_close treats +0 and -0 as equal; copying must preserve their sign.
        zeros = expected == 0
        self.assertTrue(torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros])))

    def check_plan(self, src, offsets, widths, dtype=None, require_kernel=True):
        from torch._inductor.utils import run_and_get_code

        dtype = src.dtype if dtype is None else dtype
        before = src.clone()

        table = index_table(offsets, widths, src.device)

        def fn(x):
            return passes.INDEXED_SLICE_CONCAT_TARGET(x, table, 0, table.numel(), dtype)

        expected = torch.cat([src[:, off:off + width]
                              for off, width in zip(offsets, widths)], 1).to(dtype)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        actual, codes = run_and_get_code(compiled, src)
        torch.npu.synchronize()
        self.assert_exact(actual, expected)
        self.assert_exact(src, before)
        code = "\n".join(codes)
        self.assertNotIn("= torch.ops.npu_ext.indexed_slice_concat.default(", code)
        if require_kernel:
            self.assertIn("async_compile.triton", code)
        # Reuse the compiled artifact with changed data to detect stale reads.
        storage_size = src.storage_offset() + 1 + sum(
            (size - 1) * stride for size, stride in zip(src.shape, src.stride())
        )
        storage = torch.full((max(0, storage_size),), 3, dtype=src.dtype, device=src.device)
        other = storage.as_strided(src.shape, src.stride(), src.storage_offset())
        again = compiled(other)
        torch.npu.synchronize()
        self.assert_exact(again, torch.full(expected.shape, 3, dtype=dtype, device=src.device))
        self.assert_exact(actual, expected)

    @parametrize("shape", [(1, 1), (3, 255), (4, 256), (5, 257),
                           (3, 511), (4, 512), (5, 513), (17, 645)])
    def test_tile_tails_and_last_source_column(self, shape):
        rows, cols = shape
        src = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols).npu()
        self.check_plan(src, [cols - 1, 0], [1, cols - 1] if cols > 1 else [1, 1])

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_consumer_preserves_cast_rounding(self, dtype):
        values = [1.00390625 + 2. ** -15, 1.00048828125, -1.00390625, 1.e-8]
        src = torch.tensor([values] * 5, dtype=torch.float32, device="npu")

        def fn(x):
            packed = passes.INDEXED_SLICE_CONCAT_TARGET(
                x, index_table([3, 0, 1], [1, 4, 2], x.device), 0, 7, dtype)
            return packed.float() + 1.

        expected = torch.cat([src[:, 3:4], src, src[:, 1:3]], 1).to(dtype).float() + 1.
        actual = torch.compile(fn, fullgraph=True)(src)
        torch.npu.synchronize()
        self.assert_exact(actual, expected)

    @parametrize("layout", ["row_broadcast", "column_broadcast", "external_offset",
                            "large_row_stride"])
    def test_source_layouts(self, layout):
        if layout == "row_broadcast":
            src = torch.arange(67, dtype=torch.float32, device="npu")[None, :].expand(5, 67)
        elif layout == "column_broadcast":
            src = torch.arange(5, dtype=torch.float32, device="npu")[:, None].expand(5, 67)
        elif layout == "external_offset":
            src = torch.arange(9 * 141, dtype=torch.float32, device="npu").reshape(9, 141)[2:8:2, 3:137:2]
        else:
            src = torch.arange(5 * 131072, dtype=torch.float32, device="npu").reshape(5, 131072)[:, -67:]
        self.check_plan(src, [66, 0, 1, 66], [1, 3, 65, 1])

    def test_source_conversion_preserves_rounding(self):
        src = torch.tensor([[1.00390625 + 2. ** -15, -1.00390625, 1.e-8, 0.]],
                           device="npu").expand(5, 4).clone()

        def fn(x):
            packed = passes.INDEXED_SLICE_CONCAT_TARGET(
                x.bfloat16(), index_table([3, 0, 1], [1, 4, 2], x.device), 0, 7, torch.bfloat16,
            )
            return packed.float()

        rounded = src.bfloat16()
        expected = torch.cat([rounded[:, 3:4], rounded, rounded[:, 1:3]], 1).float()
        actual = torch.compile(fn, fullgraph=True)(src)
        torch.npu.synchronize()
        self.assert_exact(actual, expected)

    @parametrize("src_dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("dst_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_cast_rounding_and_special_values(self, src_dtype, dst_dtype):
        values = [0., -0., float("inf"), -float("inf"), float("nan"),
                  1.00390625, 1.00048828125, -1.00390625, 65504.,
                  1.e-8, -1.e-8, 2. ** -14, 2. ** -126]
        src = torch.tensor([values] * 5, dtype=src_dtype, device="npu")
        self.check_plan(src, [12, 0, 4, 0], [1, 13, 5, 1], dst_dtype)

    def test_zero_rows(self):
        self.check_plan(torch.empty(0, 17, device="npu"), [16, 0], [1, 16],
                        require_kernel=False)

    def test_dynamic_rows_and_repeated_columns(self):
        from torch._inductor.utils import run_and_get_code

        def fn(src):
            return passes.INDEXED_SLICE_CONCAT_TARGET(
                src, index_table([66, 0, 3, 3], [1, 33, 9, 9], src.device), 0, 52, torch.float32
            )

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        for rows in (3, 5, 17):
            src = torch.randn(rows, 67, device="npu")
            torch._dynamo.mark_dynamic(src, 0)
            actual, codes = run_and_get_code(compiled, src)
            torch.npu.synchronize()
            self.assert_exact(actual, fn(src))

    @parametrize("count", [8, 9, 32, 33, 645])
    def test_pass_route_boundaries(self, count):
        from torch._inductor import config
        from torch._inductor.utils import run_and_get_code

        matches = []
        pass_ran = False
        expected_matches = int(
            count > 32
            or (passes.is_ascend950 and count <= config.max_pointwise_cat_inputs)
        )

        def fn(src):
            return torch.cat([torch.ops.aten.slice.Tensor(src, 1, i % 67, i % 67 + 1)
                              for i in range(count)], 1)

        def apply_pass(graph):
            nonlocal pass_ran
            pass_ran = True
            passes.multi_slice_concat_pass(graph)
            matches.extend(n for n in graph.nodes
                           if n.target is passes.INDEXED_SLICE_CONCAT_TARGET)

        src = torch.randn(5, 67, device="npu", dtype=torch.bfloat16)
        with config.patch(
            # The gather reads a table the pre_grad pass registers as a buffer.
            pre_grad_custom_pass=passes.indexed_slice_concat_table_pass,
            post_grad_custom_post_pass=apply_pass,
            fx_graph_cache=False,
            fx_graph_remote_cache=False,
        ):
            actual, codes = run_and_get_code(torch.compile(fn, fullgraph=True), src)
        torch.npu.synchronize()
        self.assert_exact(actual, fn(src))
        self.assertTrue(pass_ran, "route check requires the graph pass to run")
        self.assertEqual(len(matches), expected_matches)


if __name__ == "__main__":
    unittest.main()
