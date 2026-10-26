import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import sympy
import torch
from torch._inductor import config
from torch._inductor.dependencies import ReadWrites, StarDep
from torch._inductor.scheduler import BaseSchedulerNode, FusedSchedulerNode, Scheduler, SchedulerNode
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv

from torch_npu._inductor import config as npu_config
from torch_npu._inductor.codegen.npu_combined_scheduling import NPUCombinedScheduling
from torch_npu._inductor.codegen.scheduling import NPUNoLinearTritonScheduling, NPUTritonScheduling


class _Node(SchedulerNode):
    """Small scheduler IR using real fused-node dependency and lifetime logic."""

    def __init__(self, scheduler, name, sizes, reads=(), reduction=False):
        BaseSchedulerNode.__init__(self, scheduler)
        self.name = name
        self._sizes = sizes
        self.reduction = reduction
        self.node = None
        self.ancestors = OrderedSet()
        self.outputs = [SimpleNamespace(get_name=lambda: name)]
        self.min_order = self.max_order = 0
        self.min_input_distance = self.max_input_distance = 0
        self.read_writes = ReadWrites(
            OrderedSet(StarDep(read) for read in reads), OrderedSet([StarDep(name)]), OrderedSet()
        )
        self.unmet_dependencies = self.read_writes.reads
        self.last_usage = OrderedSet()

    def get_name(self):
        return self.name

    def is_reduction(self):
        return self.reduction

    def is_split_scan(self):
        return False

    def is_native_matmul(self):
        return False


class TestNoLinearFallbackFusion(unittest.TestCase):
    def setUp(self):
        self.graph = SimpleNamespace(
            sizevars=SizeVarAllocator(), name_to_buffer={},
            removed_buffers=OrderedSet(["already_removed"]), inplaced_to_remove=OrderedSet(),
        )
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(V.set_graph_handler(self.graph))
        stack.enter_context(mock.patch.object(npu_config, "enable_welford", False))
        stack.enter_context(config.patch({
            "triton.tiling_prevents_reduction_fusion": False,
            "triton.tiling_prevents_pointwise_fusion": False,
        }))
        self.scheduler = SimpleNamespace(
            available_buffer_names=OrderedSet(["input"]), completed_operations=OrderedSet(),
            buffer_names_to_free=OrderedSet(), mutation_real_name={}, name_to_fused_node={},
        )
        self.scheduler.topological_sort_schedule = lambda nodes: Scheduler.topological_sort_schedule(
            self.scheduler, nodes
        )
        self.backend = NPUNoLinearTritonScheduling(self.scheduler)
        self.combined = object.__new__(NPUCombinedScheduling)
        self.combined.scheduler = self.scheduler
        self.combined._nolinear_triton_scheduling = self.backend
        self.combined._triton_scheduling = mock.Mock()
        self.combined._triton_scheduling.codegen_node.side_effect = RuntimeError("force index fallback")
        self.combined._triton_scheduling.benchmark_fused_nodes.side_effect = RuntimeError("force index fallback")
        self.emitted = []
        self.freed = []
        self.backend.codegen_node = mock.Mock(side_effect=self.emit)

    def emit(self, group):
        # AllocateLine/FreeIfNotReusedLine require the outer schedule position.
        self.scheduler.nodes.index(self.scheduler.current_node)
        nodes = group.get_nodes()
        _, (numel, rnumel) = max(nodes, key=lambda n: int(n.is_reduction())).group
        # Exercise the real fallback scheduler, including its group comparisons.
        self.backend.generate_node_schedule(nodes, numel, rnumel)
        self.emitted.append(tuple(n.get_name() for n in nodes))
        self.freed.append(set(self.scheduler.buffer_names_to_free))
        self.scheduler.buffer_names_to_free.clear()

    def make_fused(self, symbolic=True, floor_div=False):
        batch, height, q = sympy.symbols("batch height q", integer=True, positive=True)
        if floor_div:
            q = FloorDiv(height - 1, 32)
        if not symbolic:
            batch, q = sympy.Integer(2), sympy.Integer(6)
        nodes = [
            _Node(self.scheduler, "producer", ([batch, 1536, q + 1, q + 1], []), ["input"]),
            _Node(self.scheduler, "reduction", ([batch, 1536], [q + 1, q + 1]), ["producer"], True),
            _Node(self.scheduler, "epilogue", ([batch, 1536], []), ["reduction"]),
        ]
        index = NPUTritonScheduling(self.scheduler)
        for i, node in enumerate(nodes):
            node.group = (torch.device("cpu"), index.group_fn(node._sizes))
            node.ancestors = OrderedSet(n.get_name() for n in nodes[:i])
            node.min_order = node.max_order = i
        # merge_loops changes the ranges but retains the original index groups.
        a = 1536 * batch
        nodes[0]._sizes = ((a * (q + 1)**2,), ())
        nodes[1]._sizes = ((a,), ((q + 1)**2,))
        nodes[2]._sizes = ((a,), ())
        fused = FusedSchedulerNode(self.scheduler, nodes)
        fused.set_last_usage(OrderedSet(["epilogue"]), {})
        self.scheduler.current_node = fused
        self.scheduler.nodes = [fused]
        self.scheduler.name_to_fused_node.update({n.get_name(): fused for n in nodes})
        self.scheduler.buffer_names_to_free.update(fused.last_usage)
        return fused, nodes

    def test_symbolic_fallback_rechecks_fusion_and_buffer_lifetimes(self):
        for floor_div in (False, True):
            with self.subTest(floor_div=floor_div):
                self.emitted.clear()
                self.freed.clear()
                fused, nodes = self.make_fused(floor_div=floor_div)
                self.combined.codegen_node(fused)
                self.assertEqual(self.emitted, [("producer",), ("reduction", "epilogue")])
                self.assertEqual(self.freed, [{"input"}, {"producer", "reduction"}])
                self.assertIs(self.scheduler.current_node, fused)
                self.assertIs(self.scheduler.name_to_fused_node["producer"], nodes[0])
                self.assertEqual(self.scheduler.completed_operations, OrderedSet(n.name for n in nodes))
                self.assertIn("producer", self.scheduler.available_buffer_names)

    def test_static_compatible_group_stays_fused(self):
        fused, _ = self.make_fused(symbolic=False)
        self.combined.codegen_node(fused)
        self.assertEqual(self.emitted, [("producer", "reduction", "epilogue")])

    def test_shared_input_and_live_output_survive_kernel_boundary(self):
        _, nodes = self.make_fused()
        nodes[-1].read_writes = nodes[-1].read_writes.with_read(StarDep("input"))
        fused = FusedSchedulerNode(self.scheduler, nodes)
        fused.set_last_usage(OrderedSet(["producer", "epilogue"]), {})
        self.scheduler.current_node = fused
        self.scheduler.nodes = [fused]
        self.scheduler.buffer_names_to_free = fused.last_usage.copy()
        self.combined.codegen_node(fused)
        self.assertEqual(self.freed, [set(), {"input", "reduction"}])

    def test_mutation_names_use_the_new_last_use(self):
        fused, _ = self.make_fused()
        self.scheduler.mutation_real_name = {"producer": "storage"}
        fused.set_last_usage(OrderedSet(["epilogue"]), self.scheduler.mutation_real_name)
        self.scheduler.buffer_names_to_free = fused.last_usage.copy()
        self.combined.codegen_node(fused)
        self.assertEqual(self.freed, [{"input"}, {"storage", "reduction"}])

    def test_backend_range_rejection_splits_matching_extent(self):
        fused, _ = self.make_fused(symbolic=False)
        with mock.patch("torch._inductor.codegen.simd.SIMDKernel.is_compatible", return_value=False):
            self.combined.codegen_node(fused)
        self.assertEqual(self.emitted, [("producer",), ("reduction", "epilogue")])

    def test_topological_order_is_restored_before_splitting(self):
        _, nodes = self.make_fused()
        self.combined._regroup_for_nolinear(nodes)
        groups = self.combined._split_nolinear_nodes(list(reversed(nodes)))
        self.assertEqual([tuple(n.name for n in g.get_nodes()) for g in groups],
                         [("producer",), ("reduction", "epilogue")])

    def test_failed_index_removed_buffers_are_restored(self):
        fused, _ = self.make_fused()

        def fail(node):
            self.graph.removed_buffers.add("producer")
            self.graph.inplaced_to_remove.add("producer")
            raise RuntimeError("failed after marking kernel-local buffers")

        self.combined._triton_scheduling.codegen_node.side_effect = fail
        self.combined.codegen_node(fused)
        self.assertEqual(self.graph.removed_buffers, OrderedSet(["already_removed"]))
        self.assertEqual(self.graph.inplaced_to_remove, OrderedSet())

    def test_index_success_does_not_regroup(self):
        fused, nodes = self.make_fused()
        groups = [node.group for node in nodes]
        self.combined._triton_scheduling.codegen_node.side_effect = None
        self.combined.codegen_node(fused)
        self.assertEqual([node.group for node in nodes], groups)
        self.backend.codegen_node.assert_not_called()

    def test_invalid_benchmark_candidate_is_rejected_and_groups_restored(self):
        _, nodes = self.make_fused()
        groups = [node.group for node in nodes]
        self.backend.benchmark_fused_nodes = mock.Mock()
        self.assertEqual(self.combined.benchmark_fused_nodes(nodes), (float("inf"), ""))
        self.backend.benchmark_fused_nodes.assert_not_called()
        self.assertEqual([node.group for node in nodes], groups)

    def test_valid_benchmark_restores_groups_even_on_error(self):
        for fails in (False, True):
            with self.subTest(fails=fails):
                _, nodes = self.make_fused(symbolic=False)
                groups = [node.group for node in nodes]
                self.backend.benchmark_fused_nodes = mock.Mock(return_value=(1.0, "kernel.py"))
                if fails:
                    self.backend.benchmark_fused_nodes.side_effect = RuntimeError("benchmark failed")
                    with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
                        self.combined.benchmark_fused_nodes(nodes)
                else:
                    self.assertEqual(self.combined.benchmark_fused_nodes(nodes), (1.0, "kernel.py"))
                self.assertEqual([node.group for node in nodes], groups)


if __name__ == "__main__":
    unittest.main()
