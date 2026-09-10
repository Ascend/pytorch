import ast
import importlib.util
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_SLOW_TESTS = os.environ.get("TORCH_NPU_FLEX_ATTENTION_SLOW_TESTS") == "1"


def _load_workspace_capacity():
    # Keep pure policy tests runnable without importing the NPU lowering stack.
    path = ROOT / "torch_npu/_inductor/kernel/flex_attention.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = {"MAX_SCAN_ROWS", "INT32_MAX"}
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.Import) and any(alias.name == "math" for alias in node.names))
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants for target in node.targets
        ))
        or (isinstance(node, ast.FunctionDef) and node.name == "fwd_mask_workspace_capacity")
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["fwd_mask_workspace_capacity"]


fwd_mask_workspace_capacity = _load_workspace_capacity()


class TestWorkspaceCapacity(unittest.TestCase):
    def test_static_workspace_policy(self):
        from itertools import product
        from types import SimpleNamespace

        tree = ast.parse((ROOT / "torch_npu/_inductor/kernel/flex_attention.py").read_text())
        assignment = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "static_workspace_path"
                for target in node.targets
            )
        )
        expression = compile(ast.Expression(assignment.value), "<workspace_policy>", "eval")
        for aot, multi_stream, dynamic in product((False, True), repeat=3):
            with self.subTest(aot=aot, multi_stream=multi_stream, dynamic=dynamic):
                namespace = dict.fromkeys(("query", "key", "value", "kv_num_blocks", "kv_indices"))
                namespace.update(
                    V=SimpleNamespace(graph=SimpleNamespace(aot_mode=aot)),
                    is_multi_stream=lambda: multi_stream,
                    _ir_has_dynamic_shape=lambda *args: dynamic,
                )
                self.assertEqual(eval(expression, namespace), not (aot or multi_stream or dynamic))

    def test_shape_bound_and_exact_budget(self):
        capacity = fwd_mask_workspace_capacity
        self.assertEqual(capacity((1, 1, 8), (1, 1, 8, 8), (128, 128), 1 << 20), 64)
        self.assertIsNone(capacity((1, 1, 8), (1, 1, 8, 8), (128, 128), (1 << 20) - 1))
        self.assertEqual(capacity((1, 1, 32), (1, 1, 32, 32), (128, 128), 16 << 20), 1024)

    def test_mask_batch_head_dimensions(self):
        self.assertEqual(fwd_mask_workspace_capacity(
            (2, 4, 8), (2, 4, 8, 8), (128, 128), 8 << 20), 512)

    def test_256_mib_budget_and_broadcasting(self):
        capacity = fwd_mask_workspace_capacity
        budget = 256 << 20
        self.assertEqual(capacity((1, 1, 4), (1, 1, 4, 4), (128, 128), budget), 16)
        self.assertEqual(capacity((4, 32, 4), (4, 32, 4, 4), (128, 128), budget), 2048)
        self.assertIsNone(capacity((4, 32, 4), (4, 32, 4, 4), (128, 128), 16 << 20))
        self.assertEqual(capacity((4, 16, 16), (4, 16, 16, 16), (128, 128), budget), 16384)
        self.assertIsNone(capacity((4, 16, 16), (4, 16, 16, 16), (128, 128), budget - 1))
        self.assertIsNone(capacity((4, 16, 16), (4, 16, 16, 17), (128, 128), budget))

    def test_disabled_invalid_and_index_limits(self):
        capacity = fwd_mask_workspace_capacity
        self.assertIsNone(capacity((1, 1, 8), (1, 1, 8, 8), (128, 128), 0))
        self.assertIsNone(capacity((1, 1, 8), (2, 1, 8, 8), (128, 128), 16 << 20))
        self.assertIsNone(capacity((1, 1, 8), (1, 1, 8, 0), (128, 128), 16 << 20))
        self.assertIsNone(capacity((1, 1, 1), (1, 1, 1, 131072), (128, 128), 1 << 32))
        self.assertIsNone(capacity((1, 1, 4097), (1, 1, 4097, 1), (64, 64), 64 << 20))


try:
    import torch
    import torch_npu
    import torch_npu._inductor
    HAS_NPU = torch.npu.is_available()
except (ImportError, RuntimeError):
    HAS_NPU = False


class _FwdMaskTestMixin:
    """Exercise identical inputs with workspace and exact allocation."""

    def setUp(self):
        wrapper_patch = torch._inductor.config.patch(cpp_wrapper=False)
        wrapper_patch.__enter__()
        self.addCleanup(wrapper_patch.__exit__, None, None, None)
        from torch_npu._inductor import config
        old_budget = config.flex_attention.fwd_mask_workspace_bytes
        config.flex_attention.fwd_mask_workspace_bytes = self.workspace_budget
        self.addCleanup(setattr, config.flex_attention, "fwd_mask_workspace_bytes", old_budget)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def assert_scalar_sync(self, compiled, *args, expected):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            compiled(*args)
            torch.npu.synchronize()
        has_scalar_sync = any(
            event.key == "aten::_local_scalar_dense" for event in prof.key_averages()
        )
        self.assertEqual(has_scalar_sync, expected)

    @unittest.skipUnless(
        RUN_SLOW_TESTS,
        "Slow cross-feature regression (CI timeout observed on exact path); "
        "set TORCH_NPU_FLEX_ATTENTION_SLOW_TESTS=1 to run",
    )
    def test_rebuilt_masks_full_capacity_and_padding(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        from torch._dynamo.testing import CompileCounterWithBackend

        q = torch.randn(2, 4, 257, 64, device="npu", dtype=torch.float16)
        k = torch.randn(1, 2, 385, 64, device="npu", dtype=torch.float16)
        v = torch.randn_like(k)
        pattern_id = torch.zeros((), dtype=torch.int32, device="npu")

        def mask(_b, _h, m, n):
            # Keep one mask graph without advanced-indexing fallback in the subgraph.
            return (
                (pattern_id == 0)
                | ((pattern_id == 1) & (m >= n))
                | ((pattern_id == 2) & (n % 2 == 0))
                | ((pattern_id == 3) & (n % 2 == 1))
                | ((pattern_id == 5) & (m >= n) & (m - n < 32))
            )

        def fn(q, k, v, bm):
            return flex_attention(q, k, v, block_mask=bm, enable_gqa=True, return_lse=True,
                                  kernel_options={"FORCE_USE_FLEX_ATTENTION": True,
                                                  "BLOCK_M": 64, "BLOCK_N": 64})

        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, dynamic=False, fullgraph=True)
        m = torch.arange(257, device="npu")[:, None]
        n = torch.arange(385, device="npu")[None, :]
        patterns = {
            "full": torch.ones(257, 385, dtype=torch.bool, device="npu"),
            "causal": m >= n,
            "even": (n % 2 == 0).expand(257, 385),
            "odd": (n % 2 == 1).expand(257, 385),
            "empty": torch.zeros(257, 385, dtype=torch.bool, device="npu"),
            "window": (m >= n) & (m - n < 32),
        }
        pattern_ids = {name: index for index, name in enumerate(patterns)}
        for name in self.mask_pattern_order:
            pattern_id.fill_(pattern_ids[name])
            # PyTorch 2.7 requires mask batch == query batch when K/V batch broadcasts.
            bm = create_block_mask(mask, 2, 1, 257, 385, device="npu")
            output, lse = compiled(q, k, v, bm)
            scores = q.float() @ k.float().repeat_interleave(2, 1).transpose(-1, -2) / 8
            scores.masked_fill_(~patterns[name], float("-inf"))
            reference = torch.softmax(scores, -1).nan_to_num() @ v.float().repeat_interleave(2, 1)
            torch.testing.assert_close(output.float(), reference, atol=3e-3, rtol=3e-3)
            torch.testing.assert_close(lse, torch.logsumexp(scores, -1), atol=3e-3, rtol=3e-3)
        self.assertEqual(counter.frame_count, 1)
        self.assert_scalar_sync(compiled, q, k, v, bm, expected=self.workspace_budget == 0)

    def test_graph_produced_counts(self):
        from torch.nn.attention.flex_attention import BlockMask, flex_attention
        from torch._dynamo.testing import CompileCounterWithBackend

        q = torch.randn(1, 1, 128, 64, device="npu", dtype=torch.float16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        base_counts = torch.ones(1, 1, 1, device="npu", dtype=torch.int32)
        indices = torch.zeros(1, 1, 1, 1, device="npu", dtype=torch.int32)
        gate = torch.ones(1, device="npu", dtype=torch.int32)

        def mask(_b, _h, _m, n):
            return (n % 2 == 0) & (gate[0] != 0)

        def fn(q, k, v, counts, indices, gate):
            bm = BlockMask.from_kv_blocks(
                counts * gate, indices, BLOCK_SIZE=(128, 128), mask_mod=mask,
            )
            return flex_attention(q, k, v, block_mask=bm,
                                  kernel_options={"FORCE_USE_FLEX_ATTENTION": True,
                                                  "BLOCK_M": 64, "BLOCK_N": 64})

        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, dynamic=False, fullgraph=True)
        for flag in (1, 0, 1):
            gate.fill_(flag)
            result = compiled(q, k, v, base_counts, indices, gate)
            scores = q.float() @ k.float().transpose(-1, -2) / 8
            scores.masked_fill_(~mask(0, 0, 0, torch.arange(128, device="npu")), float("-inf"))
            reference = torch.softmax(scores, -1).nan_to_num() @ v.float()
            torch.testing.assert_close(result.float(), reference, atol=3e-3, rtol=3e-3)
        self.assertEqual(counter.frame_count, 1)


@unittest.skipUnless(HAS_NPU, "requires NPU")
class TestFwdWorkspaceNPU(_FwdMaskTestMixin, unittest.TestCase):
    workspace_budget = 256 << 20
    mask_pattern_order = ("full", "causal", "even", "odd", "empty", "window")

    def test_batch_head_masks_and_concurrent_streams(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        q = torch.randn(2, 3, 257, 64, device="npu", dtype=torch.bfloat16)
        k = torch.randn(2, 3, 385, 64, device="npu", dtype=torch.bfloat16)
        v = torch.randn_like(k)

        def mask(b, h, m, n):
            return (m >= n) & ((n + b + h) % 3 != 0)

        bm = create_block_mask(mask, 2, 3, 257, 385, device="npu")
        def fn(q, k, v, bm):
            return flex_attention(q, k, v, block_mask=bm,
                                  kernel_options={"FORCE_USE_FLEX_ATTENTION": True,
                                                  "BLOCK_M": 64, "BLOCK_N": 64})

        compiled = torch.compile(fn, dynamic=False, fullgraph=True)
        result = compiled(q, k, v, bm)
        b, h, m, n = [torch.arange(length, device="npu").view(shape) for length, shape in
                      [(2, (2, 1, 1, 1)), (3, (1, 3, 1, 1)),
                       (257, (1, 1, 257, 1)), (385, (1, 1, 1, 385))]]
        scores = q.float() @ k.float().transpose(-1, -2) / 8
        scores.masked_fill_(~mask(b, h, m, n), float("-inf"))
        reference = torch.softmax(scores, -1).nan_to_num() @ v.float()
        torch.testing.assert_close(result.float(), reference, atol=2e-2, rtol=2e-2)
        original = torch.npu.current_stream()
        streams = [torch.npu.Stream(), torch.npu.Stream()]
        for stream in streams:
            stream.wait_stream(original)
        outputs = []
        for stream in streams:
            with torch.npu.stream(stream):
                outputs.append(compiled(q, k, v, bm))
                outputs.append(compiled(q, k, v, bm))
        for stream in streams:
            original.wait_stream(stream)
        for output in outputs:
            torch.testing.assert_close(output.float(), reference, atol=2e-2, rtol=2e-2)

    def test_prefix_scan_exact_at_row_limit(self):
        import math
        from torch._inductor.lowering import register_lowering, lowerings
        from torch_npu._inductor.kernel.flex_attention import MAX_SCAN_ROWS, _build_fwd_workspace_offsets

        @torch.library.custom_op("test_fwd_workspace::prefix", mutates_args=())
        def prefix(counts: torch.Tensor) -> torch.Tensor:
            flat = counts.flatten()
            return (flat.cumsum(0, dtype=torch.int32) - flat).view(counts.shape)

        @prefix.register_fake
        def fake_prefix(counts):
            return counts.new_empty(counts.shape)

        op = torch.ops.test_fwd_workspace.prefix.default
        @register_lowering(op, type_promotion_kind=None)
        def lower_prefix(counts):
            return _build_fwd_workspace_offsets(counts, math.prod(map(int, counts.get_size())))
        self.addCleanup(lowerings.pop, op)

        compiled = torch.compile(prefix, dynamic=False, fullgraph=True)
        for rows in (257, MAX_SCAN_ROWS):
            host = (torch.arange(rows, dtype=torch.int32) % 5).view(1, 1, rows)
            actual = compiled(host.to("npu"))
            flat = host.flatten()
            expected = flat.cumsum(0, dtype=torch.int32) - flat
            torch.testing.assert_close(actual.cpu().flatten(), expected, atol=0, rtol=0)

    def test_budget_fallback_and_fx_cache_strategy(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        from torch._inductor.utils import run_and_get_code
        from torch_npu._inductor import config

        q = torch.randn(1, 1, 256, 64, device="npu", dtype=torch.float16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        def mask(_b, _h, m, n):
            return m >= n
        bm = create_block_mask(mask, 1, 1, 256, 256, device="npu")
        def fn(q, k, v, bm):
            return flex_attention(q, k, v, block_mask=bm,
                                  kernel_options={"FORCE_USE_FLEX_ATTENTION": True,
                                                  "BLOCK_M": 64, "BLOCK_N": 64})
        reference = None
        for budget, has_sync in ((65536, False), (65535, True), (0, True)):
            config.flex_attention.fwd_mask_workspace_bytes = budget
            torch._dynamo.reset()
            compiled = torch.compile(fn, dynamic=False, fullgraph=True)
            result, codes = run_and_get_code(compiled, q, k, v, bm)
            source = "\n".join(codes)
            self.assertIn("flex_attention_compact_mapping", source)
            self.assertIn("flex_attention_fwd_mask_compact" if has_sync
                          else "flex_attention_fwd_workspace_mask_compact", source)
            self.assertNotIn("MASK_CHUNKS_PER_ROW", source)
            self.assertNotIn("fwd_global_mask", source)
            self.assertNotIn("atomic_add", source)
            if reference is None:
                reference = result
            torch.testing.assert_close(result, reference, atol=3e-3, rtol=3e-3)
            self.assert_scalar_sync(compiled, q, k, v, bm, expected=has_sync)

    @unittest.skipUnless(
        RUN_SLOW_TESTS,
        "Slow 32 MiB workspace regression; "
        "set TORCH_NPU_FLEX_ATTENTION_SLOW_TESTS=1 to run",
    )
    def test_workspace_above_16_mib(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        q = torch.randn(2, 4, 2048, 64, device="npu", dtype=torch.float16)
        k, v = torch.randn_like(q), torch.randn_like(q)

        def mask(b, h, m, n):
            return m >= n + b + h

        bm = create_block_mask(mask, 2, 4, 2048, 2048, device="npu")
        self.assertEqual(bm.kv_num_blocks.numel() * bm.kv_indices.size(-1) * 128 * 128, 32 << 20)

        def fn(q, k, v, bm):
            return flex_attention(q, k, v, block_mask=bm,
                                  kernel_options={"FORCE_USE_FLEX_ATTENTION": True,
                                                  "BLOCK_M": 128, "BLOCK_N": 128})

        compiled = torch.compile(fn, dynamic=False, fullgraph=True)
        result = compiled(q, k, v, bm)
        # Check boundary and interior rows without allocating a dense 32 MiB mask.
        rows = torch.tensor([0, 1, 5, 127, 128, 1023, 2047], device="npu")
        b = torch.arange(2, device="npu").view(2, 1, 1, 1)
        h = torch.arange(4, device="npu").view(1, 4, 1, 1)
        n = torch.arange(2048, device="npu").view(1, 1, 1, -1)
        scores = q[:, :, rows].float() @ k.float().transpose(-1, -2) / 8
        scores.masked_fill_(~mask(b, h, rows.view(1, 1, -1, 1), n), float("-inf"))
        reference = torch.softmax(scores, -1).nan_to_num() @ v.float()
        torch.testing.assert_close(result[:, :, rows].float(), reference, atol=3e-3, rtol=3e-3)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assert_scalar_sync(compiled, q, k, v, bm, expected=False)


@unittest.skipUnless(HAS_NPU, "requires NPU")
class TestFwdScanCompactNPU(_FwdMaskTestMixin, unittest.TestCase):
    workspace_budget = 0
    mask_pattern_order = ("full", "empty", "causal", "even", "odd", "window")


@unittest.skipUnless(HAS_NPU, "requires NPU")
class TestCompactScanNPU(unittest.TestCase):
    """Test the offsets kernel shared by forward exact allocation and backward."""

    def setUp(self):
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def _compile_scan(self, *, context, dynamic):
        from unittest.mock import patch
        from torch._inductor.lowering import lowerings, register_lowering
        from torch._inductor.virtualized import V
        from torch_npu._inductor.kernel.flex_attention import _build_runtime_compact_sparse_mask_offsets

        namespace = f"test_compact_scan_{context}"

        @torch.library.custom_op(f"{namespace}::scan", mutates_args=())
        def scan(counts: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            flat = counts.flatten()
            return ((flat.cumsum(0, dtype=torch.int32) - flat).view(counts.shape),
                    flat.sum(dtype=torch.int32).view(1))

        @scan.register_fake
        def fake_scan(counts, indices):
            return counts.new_empty(counts.shape), counts.new_empty((1,))

        op = getattr(torch.ops, namespace).scan.default
        if context == "forward":
            target = torch.ops.higher_order.flex_attention
        else:
            target = torch.ops.higher_order.flex_attention_backward

        @register_lowering(op, type_promotion_kind=None)
        def lower_scan(counts, indices):
            # Inductor 2.7 only permits these mutated temporaries inside the HOP.
            with patch.object(V.graph.current_node, "target", target):
                offsets, total, _ = _build_runtime_compact_sparse_mask_offsets(
                    kv_num_blocks=counts, kv_indices=indices,
                    device=counts.get_device(), context=context,
                )
            return offsets, total

        self.addCleanup(lowerings.pop, op)
        return torch.compile(scan, dynamic=dynamic, fullgraph=True)

    def test_scan_offsets_and_total_across_tiles(self):
        compiled = self._compile_scan(context="forward", dynamic=False)
        for rows in (0, 1, 4096, 4097):
            with self.subTest(rows=rows):
                # Noncontiguous input also exercises count strides at tile boundaries.
                host = (torch.arange(rows * 2, dtype=torch.int32) % 5).view(1, 1, rows * 2)[..., ::2]
                counts = (torch.arange(rows * 2, device="npu", dtype=torch.int32) % 5).view(1, 1, rows * 2)[..., ::2]
                indices = torch.zeros(1, 1, rows, 4, device="npu", dtype=torch.int32)
                offsets, total = compiled(counts, indices)
                flat = host.flatten()
                torch.testing.assert_close(offsets.cpu().flatten(), flat.cumsum(0, dtype=torch.int32) - flat,
                                           atol=0, rtol=0)
                torch.testing.assert_close(total.cpu(), flat.sum(dtype=torch.int32).view(1), atol=0, rtol=0)

    def test_dynamic_scan_strides_and_repeated_counts(self):
        compiled = self._compile_scan(context="backward", dynamic=True)
        for rows in (0, 257, 4097):
            # Broadcast batch/head strides and a non-unit row stride.
            storage = torch.zeros(1, 1, rows * 2, device="npu", dtype=torch.int32)
            counts = storage[..., ::2].expand(2, 3, rows)
            indices = torch.zeros(2, 3, rows, 5, device="npu", dtype=torch.int32)
            for pattern in ("mixed", "zero", "full", "mixed"):
                with self.subTest(rows=rows, pattern=pattern):
                    host = torch.arange(rows, dtype=torch.int32) % 6
                    if pattern == "zero":
                        host.zero_()
                    elif pattern == "full":
                        host.fill_(5)
                    storage[..., ::2].copy_(host.to("npu"))
                    offsets, total = compiled(counts, indices)
                    flat = host.view(1, 1, rows).expand(2, 3, rows).flatten()
                    torch.testing.assert_close(offsets.cpu().flatten(),
                                               flat.cumsum(0, dtype=torch.int32) - flat, atol=0, rtol=0)
                    torch.testing.assert_close(total.cpu(), flat.sum(dtype=torch.int32).view(1), atol=0, rtol=0)


@unittest.skipUnless(HAS_NPU, "requires NPU")
class TestWorkspaceCompactNPU(unittest.TestCase):
    def test_strided_autotune_metadata(self):
        from types import SimpleNamespace
        from torch._inductor.virtualized import V
        from torch_npu._inductor.kernel.flex_attention import (
            create_workspace_counts_fake, create_workspace_offsets_fake,
            create_workspace_strided_int_fake,
        )

        shape = (2, 3, 6)
        sizevars = SimpleNamespace(size_hints=lambda values, **kwargs: tuple(map(int, values)))
        with V.set_graph_handler(SimpleNamespace(sizevars=sizevars)):
            for stride in ((18, 6, 1), (36, 12, 2), (0, 12, 2)):
                node = SimpleNamespace(get_size=lambda: shape, get_stride=lambda: stride,
                                       get_dtype=lambda: torch.int32, get_device=lambda: torch.device("npu"))
                counts = create_workspace_counts_fake(node)
                self.assertEqual(counts.stride(), stride)
                self.assertTrue(bool((counts == 1).all()))
                offsets = create_workspace_offsets_fake(node).cpu().flatten()
                torch.testing.assert_close(offsets, torch.arange(36, dtype=torch.int32), atol=0, rtol=0)
                indices = create_workspace_strided_int_fake(node)
                self.assertEqual(indices.stride(), stride)
                self.assertTrue(bool((indices == 0).all()))

    def test_mapping_and_mask_only_write_live_prefix(self):
        from jinja2 import Environment, StrictUndefined
        from torch_npu._inductor.kernel.flexattention_template import (
            compute_compact_sparse_mask_mapping_kernel, compute_fwd_workspace_mask_compact,
        )

        sizes = {"KV_NUM_BLKS": ("SZ", "SH", "SQ"), "KV_IDX": ("SZ", "SH", "SQ", "COLS"),
                 "Q": (0, 0, "QLEN"), "K": (0, 0, "KLEN")}
        strides = {"KV_NUM_BLKS": ("CZ", "CH", "CQ"), "KV_IDX": ("IZ", "IH", "IQ", "IB"),
                   "Q_OFFSETS": ("OZ", "OH", "OQ")}
        constants = ("SZ SH SQ COLS QLEN KLEN CZ CH CQ IZ IH IQ IB OZ OH OQ "
                     "SPARSE_Q_BLOCK_SIZE SPARSE_KV_BLOCK_SIZE MASK_BLOCK_M MASK_BLOCK_N "
                     "NUM_Q_SUB_BLOCKS NUM_KV_SUB_BLOCKS SPARSE_MASK_STRIDE_BLK SPARSE_MASK_STRIDE_M").split()
        environment = Environment(undefined=StrictUndefined)
        environment.filters["indent_except_first"] = lambda text, n: textwrap.indent(text, "    " * n).lstrip()

        def render(name, template):
            def header(*args):
                return "def " + name + "(" + ", ".join(args) + ", " + ", ".join(
                    value + ": tl.constexpr" for value in constants) + "):"

            return "@triton.jit\n" + environment.from_string(template).render(
                def_kernel=header, size=lambda name, dim: sizes[name][dim],
                stride=lambda name, dim=None: strides[name][dim] if dim is not None else ", ".join(strides[name]),
                modification=lambda **kwargs: "mask_mod_output = ((m + 3 * off_z + off_h) >= n) & ((m + n) % 3 != 0)",
            ).lstrip()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace_compact_payload.py"
            path.write_text("import triton\nimport triton.language as tl\n"
                            + render("mapping_kernel", compute_compact_sparse_mask_mapping_kernel)
                            + "\n" + render("mask_kernel", compute_fwd_workspace_mask_compact))
            module_spec = importlib.util.spec_from_file_location("_workspace_compact_payload", path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = module
            self.addCleanup(sys.modules.pop, module_spec.name, None)
            module_spec.loader.exec_module(module)

            # Cover every input pattern and tiling without compiling their full product.
            cases = (
                ([0, 0, 0], 1, 1, (64, 64), 7),
                ([407], 1, 1, (64, 64), 56),
                ([1] * 406 + [407], 1, 1, (64, 32), 7),
                ([0, 2, 0, 0, 3, 0], 1, 1, (32, 32), 56),
                ([1] + [0] * 1024 + [19], 1, 1, (64, 64), 7),
                ([1, 2, 0, 1, 0, 2], 2, 3, (32, 32), 56),
            )
            for values, batches, heads, tile, programs in cases:
                tile_m, tile_n = tile
                rows, columns = len(values), max([1] + values)
                host = torch.tensor(values, dtype=torch.int32).view(1, 1, rows).expand(batches, heads, rows)
                counts = torch.empty(batches, heads, rows * 2, device="npu", dtype=torch.int32)[..., ::2]
                counts.copy_(host)
                indices = torch.empty(batches, heads, rows * 2, columns * 2, device="npu", dtype=torch.int32)[..., ::2, ::2]
                block_ids = torch.arange(columns - 1, -1, -1, dtype=torch.int32)
                indices.copy_(block_ids.to("npu"))
                flat_counts = host.flatten()
                total = int(flat_counts.sum())
                offsets = (flat_counts.cumsum(0, dtype=torch.int32) - flat_counts).view(host.shape).to("npu")
                capacity = total + 257
                block = 64
                qlen, klen = rows * block - 7, columns * block - 11
                row_ids = torch.repeat_interleave(torch.arange(flat_counts.numel()), flat_counts.long())
                flat_offsets = flat_counts.cumsum(0) - flat_counts
                positions = torch.arange(total) - flat_offsets[row_ids]
                m = (row_ids % rows * block)[:, None, None] + torch.arange(block)[None, :, None]
                n = block_ids[positions][:, None, None] * block + torch.arange(block)[None, None, :]
                h = (row_ids // rows % heads)[:, None, None]
                b = (row_ids // (rows * heads))[:, None, None]
                expected = ((m + 3 * b + h >= n) & ((m + n) % 3 != 0) & (m < qlen) & (n < klen)).to(torch.int8)
                flat_to_row = torch.full((capacity,), -31337, device="npu", dtype=torch.int32)
                flat_to_blk = torch.full_like(flat_to_row, -31337)
                args = (batches, heads, rows, columns, qlen, klen, *counts.stride(), *indices.stride(),
                        *offsets.stride(), block, block, tile_m, tile_n, block // tile_m, block // tile_n,
                        block * block, block)
                module.mapping_kernel[(min(batches * heads * rows, 56),)](
                    flat_to_row, flat_to_blk, offsets, counts, *args, num_warps=4, num_stages=1,
                )
                torch.testing.assert_close(flat_to_row[:total].cpu().long(), row_ids, atol=0, rtol=0)
                torch.testing.assert_close(flat_to_blk[:total].cpu().long(), positions, atol=0, rtol=0)
                self.assertTrue(bool((flat_to_row[total:] == -31337).all()))
                self.assertTrue(bool((flat_to_blk[total:] == -31337).all()))
                with self.subTest(rows=rows, batches=batches, heads=heads, tile=(tile_m, tile_n), programs=programs):
                    payload = torch.full((capacity, block, block), 7, device="npu", dtype=torch.int8)
                    module.mask_kernel[(programs,)](
                        payload, flat_to_row, flat_to_blk, payload, payload, indices, offsets, counts,
                        *args, num_warps=4, num_stages=1,
                    )
                    actual = payload.cpu()
                    torch.testing.assert_close(actual[:total], expected, atol=0, rtol=0)
                    self.assertTrue(bool((actual[total:] == 7).all()))


if __name__ == "__main__":
    unittest.main()
