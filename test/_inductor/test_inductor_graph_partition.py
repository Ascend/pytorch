import os

# Some ops (e.g. aten.matmul_backward) only get their NPU meta registration when
# compatible impl mode is enabled. This env var is read at torch_npu import time,
# so it must be set before importing torch_npu.
os.environ.setdefault("TORCH_NPU_USE_COMPATIBLE_IMPL", "1")

import contextlib
import gc
import math
import re
import sys
import unittest
import warnings
import weakref
from io import StringIO

import torch
import torch.nn as nn
import torch._dynamo.config as dynamo_config
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.utils import run_and_get_code
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.logging_utils import logs_to_string
from torch.utils._python_dispatch import TorchDispatchMode

import torch_npu  # noqa: F401
from torch_npu.npu._graph_tree import get_container

TEST_NPU = torch.npu.is_available()
aten = torch.ops.aten

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_num_partitions(code):
    """Get the number of graph partitions from generated code."""
    code = "".join(code)
    found = re.search(r"partitions=\[(.*)\]", code)
    assert found is not None, "Could not find partitions in generated code"
    partitions = found.group(1)
    return len([p for p in partitions.split(",") if p])


class capture_stderr(list):
    """Replace sys.stderr with a temporary StringIO."""

    def __enter__(self):
        self.sys_stderr = sys.stderr
        self.stringio = StringIO()
        sys.stderr = self.stringio
        return self

    def __exit__(self, *args):
        self.append(str(self.stringio.getvalue()))
        del self.stringio
        sys.stderr = self.sys_stderr


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------


class TestCase(InductorTestCase):
    device = "npu"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,
                    "implicit_fallbacks": False,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()


class TestGraphPartitionNPU(TestCase):
    """Tests that graph partition works end-to-end with NPU graph trees.
    Many tests verify npugraph tree state (partition count, graph id, etc.).
    """

    def setUp(self):
        super().setUp()
        self.graph_stack = contextlib.ExitStack()
        self.graph_stack.enter_context(
            config.patch(
                {
                    "triton.cudagraphs": True,
                    "triton.cudagraph_trees": True,
                }
            )
        )
        self.graph_stack.enter_context(
            dynamo_config.patch(automatic_dynamic_shapes=True)
        )
        self.device_idx = torch.rand([0], device="npu").device.index
        warnings.filterwarnings("ignore")

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()
        gc.collect()
        torch.npu.empty_cache()
        self.graph_stack.close()
        # NPU's TreeManagerContainer holds a strong reference to the tree manager;
        # explicitly clear it so each test sees a fresh manager state under
        # pytest single-process execution.
        from torch_npu.npu._graph_tree import reset_npugraph_trees
        reset_npugraph_trees()
        warnings.resetwarnings()

    def get_manager(self, device_index=None):
        return get_container(
            device_index if device_index else self.device_idx
        ).tree_manager

    # -----------------------------------------------------------------------
    # Basic partition tests
    # -----------------------------------------------------------------------

    def test_graph_partition_simple(self):
        def f(x, y):
            x1 = x + 1
            y1 = y + 1
            y_cpu = y1.cpu() + 1
            z = x @ y
            return x1 + y1 + z + y_cpu.to("npu")

        x, y = [torch.ones(2, 2, device="npu") for _ in range(2)]
        x_cloned, y_cloned = [tmp.clone() for tmp in [x, y]]
        eager_out = f(x, y)

        f_compiled = torch.compile(f)
        compiled_out = f_compiled(x_cloned, y_cloned)
        self.assertEqual(eager_out, compiled_out)

        _, code = run_and_get_code(f_compiled, x_cloned, y_cloned)

        if not config.cpp_wrapper:
            FileCheck().check("def partition_0(args):").check(
                "recursively_apply_fns = runner.recursively_apply_fns"
            ).run(code[0])

    @config.patch("graph_partition", True)
    def test_graph_partition_view_fallback(self):
        def f(x):
            y = x + 1
            z = torch.ops.aten.view.dtype(y, torch.float8_e4m3fn)
            z_cpu = z.cpu()
            u_npu = z_cpu.npu()
            return u_npu

        compiled_f = torch.compile(f, mode="reduce-overhead")

        for _ in range(3):
            x = torch.ones(2, dtype=torch.int32, device="npu")
            eager_out = f(x)
            compiled_out = compiled_f(x)
            # NPU aclnnIsClose does not support float8, compare via int8 view
            self.assertEqual(
                eager_out.view(torch.int8), compiled_out.view(torch.int8)
            )

    @config.patch("graph_partition", True)
    def test_graph_partition_log_message(self):
        def foo(x, y):
            return (x + 1, y + 2)

        foo = torch.compile(foo, mode="reduce-overhead")

        log_stream, ctx = logs_to_string("torch._inductor.utils", "perf_hints")
        with ctx():
            foo(torch.ones([10], device="npu"), torch.ones([20]))

        FileCheck().check_count(
            "cudagraph partition due to non gpu ops",
            1,
            exactly=True,
        ).check_count(
            "cudagraph partition into 2 partitions", 1, exactly=True
        ).run(log_stream.getvalue())

        log_stream, ctx = logs_to_string("torch_npu.npugraph", "cudagraphs")
        with ctx():
            # trigger recording
            foo(torch.ones([10], device="npu"), torch.ones([20]))
            foo(torch.ones([10], device="npu"), torch.ones([20]))

        FileCheck().check_count(
            "[NPUGRAPH-TREE][Node][Record] function=0, graph=0",
            1,
            exactly=True,
        ).run(log_stream.getvalue())

    # -----------------------------------------------------------------------
    # CPU scalar tests
    # -----------------------------------------------------------------------

    @config.patch("graph_partition", True)
    def test_graph_partition_cpu_scalar_device_put(self):
        @torch.compile(mode="reduce-overhead")
        def foo(x):
            y = x.to("npu")
            z = y.to("cpu")
            return z

        x = torch.tensor(1)
        for _ in range(3):
            foo(x)

        self.assertEqual(x, torch.tensor(1, device="cpu"))

    @config.patch("graph_partition", True)
    def test_graph_partition_forward_with_skipped_cudagraphed_backward(self):
        @torch.compile(mode="reduce-overhead")
        def foo(x):
            return x * x * x

        for _ in range(3):
            inp = torch.rand([20, 20], device="npu", requires_grad=True)
            out = foo(inp)

            with config.patch(always_complex_memory_overlap_TESTING_ONLY=True):
                back_inp = torch.empty_strided([20, 20], [0, 1], device="npu")
                out.backward(back_inp)

        # we should not have npugraph'd the backwards
        new_id = self.get_manager().new_graph_id().id
        self.assertEqual(new_id, 1)

        self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

    @config.patch("graph_partition", True)
    def test_graph_partition_dynamic_shapes(self):
        def foo(x):
            return x + 1

        compiled_foo = torch.compile(foo, mode="reduce-overhead", fullgraph=True)

        for input_shape in range(1, 4):
            for _ in range(3):
                compiled_foo(torch.randn(input_shape, device="npu"))

        # 3 npugraphs for 3 input shapes
        self.assertEqual(self.get_manager().new_graph_id().id, 3)

    @config.patch("graph_partition", True)
    def test_graph_partition_condition_op(self):
        def f(p, b):
            def true_fn(x):
                return torch.cos(x)

            def false_fn(x):
                return torch.sin(x)

            return torch.cond(p, true_fn, false_fn, [b])

        compiled_f = torch.compile(f)

        # static shape
        p = torch.tensor([True], device="npu")
        a = torch.ones([2, 3], device="npu")
        eager_out = f(p, a)
        compiled_out = compiled_f(p, a)
        self.assertEqual(eager_out, compiled_out)

        # dynamic shape with backed symint
        p = torch.tensor([True], device="npu")
        a = torch.ones([4, 5], device="npu")
        eager_out = f(p, a)
        compiled_out = compiled_f(p, a)
        self.assertEqual(eager_out, compiled_out)

    @config.patch("graph_partition", True)
    def test_graph_partition_reorder_cpu_and_gpu(self):
        def f(x_npu, y_cpu, z_npu, weight_npu, weight_cpu):
            x_npu0 = x_npu + 1
            x_npu1 = x_npu0 @ weight_npu
            x_npu2 = 2 * (x_npu1 + x_npu)

            y_cpu0 = y_cpu + 1
            y_cpu1 = y_cpu0 @ weight_cpu

            z_npu0 = z_npu + 1
            z_npu1 = z_npu0 @ weight_npu
            z_npu2 = 2 * (z_npu1 + z_npu)

            return x_npu2, y_cpu1, z_npu2

        x_npu = torch.randn(3, 3, device="npu")
        y_cpu = torch.randn(3, 3, device="cpu")
        z_npu = torch.randn(3, 3, device="npu")
        weight_npu = torch.randn(3, 3, device="npu")
        weight_cpu = torch.randn(3, 3, device="cpu")

        eager_out = f(x_npu, y_cpu, z_npu, weight_npu, weight_cpu)

        compiled_f = torch.compile(f, mode="reduce-overhead")
        for _ in range(3):
            compiled_out = compiled_f(x_npu, y_cpu, z_npu, weight_npu, weight_cpu)
            self.assertEqual(eager_out, compiled_out)

        # reorder merges ops on npu into 1 graph partition
        self.assertEqual(self.get_manager().new_graph_id().id, 1)

    @config.patch(implicit_fallbacks=True)
    @config.patch("graph_partition", True)
    def test_graph_partition_custom_op(self):
        @torch.library.custom_op(
            "mylib::movement_npu",
            mutates_args=(),
            tags=(torch._C.Tag.cudagraph_unsafe,),
        )
        def movement(pic: torch.Tensor) -> torch.Tensor:
            img = pic.cpu()
            cropped_img = (img + 1) * 2
            return cropped_img.npu() / 255.0

        @movement.register_fake
        def _(pic):
            return torch.empty_like(pic)

        @torch.library.custom_op(
            "mylib::modify_npu",
            mutates_args=(),
            tags=(torch._C.Tag.cudagraph_unsafe,),
        )
        def modify(pic: torch.Tensor) -> torch.Tensor:
            pic1 = pic + 1
            pic1_cpu = (pic1.cpu() + 1) * 2
            return pic1_cpu.npu() + pic

        @modify.register_fake
        def _(pic):
            return torch.empty_like(pic)

        @torch.library.custom_op("mylib::transform_npu", mutates_args=())
        def transform(pic: torch.Tensor) -> torch.Tensor:
            return (pic + 1) * 2

        @transform.register_fake
        def _(pic):
            return torch.empty_like(pic)

        img = torch.randn(3, 64, 64, device="npu")

        def f(img):
            x = (img + 10) * 2
            y = movement(x)
            z = y + 1
            u = transform(z)
            v = 2 * u + 1
            out = modify(v)
            return out + 1

        compiled_f = torch.compile(f, fullgraph=True)

        eager_out = f(img)
        compiled_out = compiled_f(img)

        self.assertEqual(eager_out, compiled_out)

        compiled_f = torch.compile(f, mode="reduce-overhead", fullgraph=True)

        eager_out = f(img)

        for _ in range(3):
            compiled_out = compiled_f(img)
            self.assertEqual(eager_out, compiled_out)

        # splitting on 2 custom gives 3 npugraphs
        self.assertEqual(self.get_manager().new_graph_id().id, 3)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
