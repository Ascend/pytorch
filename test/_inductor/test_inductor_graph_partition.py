# Owner(s): ["module: inductor"]

import os

# Some ops (e.g. aten.matmul_backward) only get their NPU meta registration when
# compatible impl mode is enabled. This env var is read at torch_npu import time,
# so it must be set before importing torch_npu.
os.environ.setdefault("TORCH_NPU_USE_COMPATIBLE_IMPL", "1")

import contextlib
import gc
import re
import unittest
import warnings
import weakref

import torch

# torch_npu must be imported before torch.testing._internal.inductor_utils (or
# anything that may transitively import it): importing torch_npu replaces
# torch.utils._triton.has_triton with an NPU-aware implementation, while
# inductor_utils freezes HAS_TRITON = has_triton() at import time. With the
# wrong order @requires_triton() would silently skip every test on NPU.
import torch_npu  # noqa: F401

import torch._dynamo.config as dynamo_config
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.inductor_utils import requires_triton
from torch.utils._python_dispatch import TorchDispatchMode

from torch_npu._compat.version import CURRENT_VERSION
from torch_npu.npu._graph_tree import get_container

try:
    from version_mark import runIfVersion
except ImportError:
    # version_mark.py lives in the repo's test/ directory and is only reachable
    # when running from the repo (CI always does). Fall back to a no-op so this
    # file can also run standalone (e.g. copied to a test server). The CI-side
    # file-level filter matches @runIfVersion decorators syntactically, so this
    # fallback does not affect CI version routing.
    def runIfVersion(min=None, max=None):
        def decorator(target):
            return target

        return decorator

TEST_NPU = torch.npu.is_available()
# COMPAT(< 2.14): upstream Inductor does not pass kernel_free_cudagraph through
# cudagraphify. The graph-tree compatibility shim keeps the attribute for
# imports, but there is no producer for the flag on those versions.
KERNEL_FREE_CUDAGRAPH_SUPPORTED = CURRENT_VERSION >= (2, 14)
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


# ===========================================================================
# Graph Partition Tests — Codegen correctness
# (ported from test_torchinductor.py, device-agnostic via self.device)
# ===========================================================================


@runIfVersion(max="2.13")
@requires_triton()
@unittest.skipIf(not TEST_NPU, "requires NPU")
class TestGraphPartitionCodegen(TestCase):
    """Tests that graph partition generates correct code and produces correct
    results. These tests do NOT verify npugraph tree state."""

    @config.patch("graph_partition", True)
    def test_graph_partition_refcount(self):
        # Trigger NPU backend registration: compile_fx_inner is a low-level
        # entry that bypasses dynamo's lazy loading of torch_npu._inductor,
        # so without this warmup get_wrapper_codegen_for_device('npu') returns
        # None and init_wrapper_code asserts "Device npu not supported".
        @torch.compile
        def _warmup(x):
            return x + 1

        _warmup(torch.ones(2, device=self.device))

        contexts = [
            contextlib.nullcontext,
            lambda: config.patch({"triton.cudagraphs": True}),
        ]

        for context in contexts:
            with context():
                inps = [
                    torch.rand([5, 5]).to(self.device),
                    torch.rand([5, 5]).to(self.device),
                ]
                inp_refs = [weakref.ref(inp) for inp in inps]

                def fn(x, y):
                    a = x + y
                    return (a @ a,)

                fn_fx = make_fx(fn)(inps[0], inps[1])
                fn_compiled = compile_fx_inner(fn_fx, inps)

                matmul_seen = False

                class TestRefMode(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        kwargs = kwargs if kwargs else {}

                        nonlocal inps
                        nonlocal inp_refs
                        nonlocal matmul_seen

                        gc.collect()
                        if func is aten.mm.out:
                            matmul_seen = True
                            assert len(inps) == 0
                            assert inp_refs[0]() is None
                            assert inp_refs[1]() is None

                        return func(*args, **kwargs)

                with TestRefMode():
                    fn_compiled(inps)

                assert matmul_seen

                # do an extra run to make sure we are deallocating on warmup and record
                inps.extend(
                    [
                        torch.rand([5, 5]).to(self.device),
                        torch.rand([5, 5]).to(self.device),
                    ]
                )
                inp_refs.extend([weakref.ref(inp) for inp in inps])
                matmul_seen = False

                with TestRefMode():
                    fn_compiled(inps)

                assert matmul_seen
                assert len(inps) == 0


# ===========================================================================
# Graph Partition Tests — Runtime & NPU Graph tree integration
# (ported from test_cudagraph_trees.py CudaGraphTreeTests)
# ===========================================================================


class GraphPartitionTestBase(TestCase):
    """Shared graph tree setup for NPU graph partition tests."""

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


@runIfVersion(max="2.13")
@requires_triton()
@unittest.skipIf(not TEST_NPU, "requires NPU")
class TestGraphPartitionNPU(GraphPartitionTestBase):
    """Tests that graph partition works end-to-end with NPU graph trees.
    Many tests verify npugraph tree state (partition count, graph id, etc.).
    """

    # -----------------------------------------------------------------------
    # Basic partition tests
    # -----------------------------------------------------------------------

    def test_graph_partition(self):
        def f(x, y):
            x1 = x + 1
            y1 = y + 1
            y_cpu = y1.cpu() + 1
            z = x @ y
            return x1 + y1 + z + y_cpu.npu()

        x, y = [torch.randn(2, 2, device="npu") for _ in range(2)]
        x_cloned, y_cloned = [tmp.clone() for tmp in [x, y]]
        eager_out = f(x, y)

        f_compiled = torch.compile(f, mode="reduce-overhead")

        for _ in range(5):
            compiled_out = f_compiled(x_cloned, y_cloned)
            self.assertEqual(eager_out, compiled_out)

        # 2 graph partitions lead to 2 npugraphs
        self.assertEqual(self.get_manager().new_graph_id().id, 2)

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

    # -----------------------------------------------------------------------
    # CPU scalar tests
    # -----------------------------------------------------------------------

    @config.patch("graph_partition", True)
    def test_graph_partition_cpu_scalar1(self):
        def f(x, y):
            return x + y

        compiled_f = torch.compile(f, mode="reduce-overhead")

        inputs = (torch.ones(2, 2, device="npu"), torch.ones((), device="cpu"))
        for i in range(3):
            if i == 0:
                _, code = run_and_get_code(compiled_f, *inputs)
                FileCheck().check_count(".copy_", 1, exactly=True).run(code[0])
            else:
                compiled_f(*inputs)
        self.assertEqual(compiled_f(*inputs), f(*inputs))
        self.assertEqual(self.get_manager().new_graph_id().id, 1)

    # -----------------------------------------------------------------------
    # reduce-overhead mode effectiveness
    # -----------------------------------------------------------------------

    @config.patch("graph_partition", True)
    @config.patch("triton.cudagraphs", False)
    def test_graph_partition_reduce_overhead_mode_effectiveness(self):
        # test that `mode="reduce-overhead"` still controls whether
        # npugraph is applied. i.e., npugraph is not applied when
        # mode="default".
        def f(x, y):
            x1 = x + 1
            y1 = y + 1
            y_cpu = y1.cpu() + 1
            z = x @ y
            return x1 + y1 + z + y_cpu.npu()

        x, y = [torch.randn(2, 2, device="npu") for _ in range(2)]

        f_compiled = torch.compile(f)
        for _ in range(5):
            _out = f_compiled(x, y)
        self.assertEqual(self.get_manager() is None, True)

    # -----------------------------------------------------------------------
    # Dynamic shapes
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Reorder tests
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Custom ops
    # -----------------------------------------------------------------------

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

    @config.patch(implicit_fallbacks=True)
    @config.patch("graph_partition", True)
    def test_graph_partition_custom_rule(self):
        @torch.library.custom_op("mylib::bar_npu", mutates_args=())
        def bar(x: torch.Tensor, flag: int) -> torch.Tensor:
            return x.clone()

        @bar.register_fake
        def _(x, flag):
            return x.clone()

        def f(x, flag):
            x = x + 1
            x = bar(x, flag)
            x = x + 1
            return x

        x = torch.randn(2, device="npu")
        f_compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
        _, code = run_and_get_code(f_compiled, x, True)
        num_partitions = get_num_partitions(code)
        self.assertEqual(num_partitions, 1)

        @torch.library.custom_op("mylib::baz_npu", mutates_args=())
        def baz(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @baz.register_fake
        def _(x):
            return x.clone()

        # custom_should_partition_ops takes effect which lead to 2 partitions
        torch._inductor.config.custom_should_partition_ops = ["mylib::baz_npu"]

        def f2(x):
            x = x + 1
            x = baz(x)
            x = x + 1
            return x

        f2_compiled = torch.compile(f2, mode="reduce-overhead", fullgraph=True)
        _, code = run_and_get_code(f2_compiled, x)
        num_partitions = get_num_partitions(code)
        self.assertEqual(num_partitions, 2)

        # update the config should NOT force recompile
        torch._inductor.config.custom_should_partition_ops = []
        with torch.compiler.set_stance("fail_on_recompile"):
            f2_compiled(x)

        # run_and_get_code forces recompile. Now we should cache miss, recompile, and
        # only have 1 partition.
        _, code = run_and_get_code(f2_compiled, x)
        num_partitions = get_num_partitions(code)
        self.assertEqual(num_partitions, 1)

        # test that op_overload name takes effect which lead to 2 partitions
        torch._inductor.config.custom_should_partition_ops = [
            "mylib::baz_npu.default"
        ]

        f2_compiled = torch.compile(f2, mode="reduce-overhead", fullgraph=True)
        _, code = run_and_get_code(f2_compiled, x)
        num_partitions = get_num_partitions(code)
        self.assertEqual(num_partitions, 2)


@requires_triton()
@unittest.skipIf(not TEST_NPU, "requires NPU")
class TestGraphPartitionE2E(GraphPartitionTestBase):
    """End-to-end graph partition coverage for current PyTorch versions."""

    @runIfVersion(min="2.14")
    @unittest.skipUnless(
        KERNEL_FREE_CUDAGRAPH_SUPPORTED,
        "kernel_free_cudagraph is only forwarded by PyTorch >= 2.14",
    )
    @config.patch("graph_partition", False)
    def test_kernel_free_view_only_graph(self):
        # View-only graphs have no device kernels. Keep the graph unpartitioned
        # so it reaches the kernel-free tree-node path directly; the node must
        # still execute in the private pool, but must not create or replay an
        # empty NPU graph.
        def f(x):
            return x.reshape(1, -1, 4)

        x = torch.randn(6, 4, device="npu")
        compiled_f = torch.compile(f, mode="reduce-overhead", fullgraph=True)

        for _ in range(3):
            self.assertEqual(compiled_f(x), f(x))

        manager = self.get_manager()
        self.assertIsNotNone(manager)
        nodes = list(manager.get_roots())
        self.assertEqual(len(nodes), 1)
        self.assertTrue(nodes[0].wrapped_function.kernel_free_cudagraph)
        self.assertIsNone(nodes[0].graph)

    @config.patch("graph_partition", True)
    @dynamo_config.patch(capture_scalar_outputs=True)
    def test_graph_partition_cpu_scalar_item(self):
        # The CPU scalar partition contains a host-side .item() and must not
        # be capture-unsafe or change the result.
        def f(x):
            y = x + 1
            scalar = y.item()
            return x + y + scalar

        x = torch.tensor(1, device="npu")
        compiled_f = torch.compile(f, mode="reduce-overhead")
        self.assertEqual(compiled_f(x), f(x))

    @config.patch("graph_partition", True)
    def test_graph_partition_forward_backward(self):
        # A real partitioned model must capture both forward partitions and
        # the backward partition without losing accuracy.
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                x1 = x + 1
                y1 = x + 2
                y_cpu = y1.cpu() + 1
                z = x @ y1
                inp = x1 + y1 + z + y_cpu.npu()
                return self.linear(inp)

        model = Mod().npu()
        input_data = torch.randn(16, 16, device="npu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        compiled_model = torch.compile(model, mode="reduce-overhead")

        for _ in range(5):
            output = compiled_model(input_data)
            target = torch.randint(0, 10, (16,), device="npu")
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(loss))

        # 2 forward partitions and 1 backward partition.
        self.assertEqual(self.get_manager().new_graph_id().id, 3)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
