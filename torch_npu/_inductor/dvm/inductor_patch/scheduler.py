"""Scheduler hooks reused by DVM and legacy MLIR.

Backends explicitly install dependency hooks before lowering and buffer-size
hooks afterwards. Importing this module does not install either group.
"""

import sympy
import collections
from collections import Counter
import functools
import itertools
from typing import (
    Dict,
    List,
    Set,
    Union,
    Optional,
    Sequence,
)

from torch._inductor import ir
from torch._inductor import scheduler
from torch._inductor.scheduler import (
    Dep,
    WeakDep,
    SchedulerBuffer,
    ForeachKernelSchedulerNode,
    WhyNoFuse,
    MemoryDep,
    SchedulerNode,
    BaseSchedulerNode,
    FusedSchedulerNode,
    NopKernelSchedulerNode,
    ExternKernelSchedulerNode,
    OutputNode,
    MultiOutput,
    MultiOutputLayout,
    OrderedSet,
    get_dtype_size,
    sympy_product,
    V,
)

def _npu_get_read_write_buffers_sizes(self) -> int:
    """
    Counting the number of bytes accessed for a kernel is
    surprisingly tricky. In particular, there is a differentiation
    between 'theoretical' memory accesses and practical memory
    accesses. For example, a layernorm kernel may actually access an
    input 3 times, but in theory, it only needs to access its input
    once (and may be optimized to do so through say, persistent
    reductions)

    Another example is that even though a buffer is passed in, we may
    not access the entire buffer. This may occur if we are accessing
    a slice of the buffer. Another tricky case is for indirect
    indexing, where the amount of bytes accessed depends on the
    values of the input.

    What this function aims to compute is the memory accesses for
    worst-case inputs, best-case optimization. What this means is
    that for each buffer we compute the amount of potential accesses in two ways and take the minimum.

    1. Numel in ranges multiplied by number of deps the buffer has
    2. The buffer size
    """
    if isinstance(self, NopKernelSchedulerNode):
        return 0
    if isinstance(self, ExternKernelSchedulerNode) and isinstance(
        self.node, MultiOutput
    ):
        # todo: Calculate this - it's kinda annoying.
        return 0

    def try_size_hint(s: sympy.Expr) -> int:
        return V.graph.sizevars.optimization_hint(s, fallback=0)

    if isinstance(self, SchedulerNode):
        node_numel = try_size_hint(
            sympy_product(self.get_ranges()[0])
            * sympy_product(self.get_ranges()[1]),
        )
    else:
        node_numel = int(1e9)
    buf_accesses = collections.defaultdict(list)
    for dep in self.read_writes.reads | self.read_writes.writes:
        buf_accesses[dep.name].append(dep)

    reads = OrderedSet(dep.name for dep in self.read_writes.reads)
    writes = OrderedSet(dep.name for dep in self.read_writes.writes)

    def is_materialized(buf: str, snodes: Sequence[BaseSchedulerNode]) -> bool:
        users = self.scheduler.name_to_buf[buf].users
        buf_uses = OrderedSet(user.node for user in users)
        return len(buf_uses - OrderedSet(snodes)) > 0

    if isinstance(self, FusedSchedulerNode):
        removed_buffers = OrderedSet(
            dep for dep in writes if not is_materialized(dep, self.snodes)
        )
        writes = writes - removed_buffers
        reads = reads - removed_buffers
    node_bytes = 0

    for buf_name in reads | writes:
        buf_accessed_elems = sum(node_numel for dep in buf_accesses[buf_name])
        buf: Union[ir.Buffer, ir.TensorBox]
        if buf_name in V.graph.name_to_buffer:
            buf = V.graph.name_to_buffer[buf_name]
        elif buf_name in V.graph.graph_inputs:
            buf = V.graph.graph_inputs[buf_name]
        else:
            continue

        def get_buf_bytes(buf: Optional[Union[ir.Buffer, ir.TensorBox]]) -> int:
            if not buf:
                return 0
            # Kind of a lazy way to get the MultiOutput nodes corresponding to
            # a MultiOutputLayout
            if isinstance(buf.layout, MultiOutputLayout):
                users = self.scheduler.name_to_buf[buf.get_name()].users
                tot = 0
                for user in users:
                    # Custom ops can return a mixed output of tensor and ints.
                    # This could happen when the custom op return symints,
                    assert isinstance(user.node, (BaseSchedulerNode, OutputNode))
                    if isinstance(user.node, BaseSchedulerNode):
                        if isinstance(user.node.node, MultiOutput):
                            for sched_buf in user.node.get_outputs():
                                tot += get_buf_bytes(sched_buf.node)
                        else:
                            # Buf is a MultiOutputLayout but not all of its
                            # users are MultiOutputs...
                            # TODO: Figure out what's going on
                            return 0
                return tot
            elif isinstance(buf.layout, ir.NoneLayout):
                return sum(
                    get_buf_bytes(V.graph.get_buffer(mut_name))
                    for mut_name in buf.get_mutation_names()
                )
            else:
                buf_elems = try_size_hint(sympy_product(buf.get_size()))
                return get_dtype_size(buf.get_dtype()) * min(
                    buf_accessed_elems, buf_elems
                )

        node_bytes += get_buf_bytes(buf)

    return node_bytes


def _patch_scheduler():
    scheduler.BaseSchedulerNode.get_read_write_buffers_sizes = _npu_get_read_write_buffers_sizes


def used_or_aliased_buffer_names(node) -> Set[str]:
    used_names: OrderedSet[str] = OrderedSet()
    if isinstance(node, (SchedulerNode, FusedSchedulerNode)) and not isinstance(
        node, ForeachKernelSchedulerNode
    ):
        snodes = [node] if isinstance(node, SchedulerNode) else node.snodes
        traced_graphs = [
            getattr(getattr(snode.node, "data", None), "traced_graph", None)
            for snode in snodes
        ]
        if all(traced_graph is not None for traced_graph in traced_graphs):
            for snode, traced_graph in zip(snodes, traced_graphs):
                used_names = used_names.union(traced_graph.get_placeholder_names())
                used_names.add(snode.node.get_name())
            return used_names

    # TritonTemplateBuffer and other non-MLIR scheduler nodes do not carry a
    # TracedGraph.  Their scheduler dependencies are the source of truth.
    deps = [
        dep.name
        for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes)
    ]
    while len(deps) > 0:
        dep = deps.pop()
        used_names.add(dep)
        if V.graph.name_to_buffer.get(dep):
            aliases = V.graph.name_to_buffer[dep].get_inputs_that_alias_output()
            for alias in aliases:
                if alias not in used_names:
                    deps.append(alias)
    return used_names


def set_last_usage(
    node: BaseSchedulerNode, future_used_buffers: Set[str], mutation_real_name: Dict[str, str]
):
    used_buffers = used_or_aliased_buffer_names(node)
    used_buffers = OrderedSet(mutation_real_name.get(k, k) for k in used_buffers)
    node.last_usage = used_buffers - future_used_buffers


def wrap_scheduler_codegen(fn):
    @functools.wraps(fn)
    def npu_sheduler_codegen(self, *args, **kwargs):
        future_used_buffers = set()
        for node_name in V.graph.get_output_names():
            future_used_buffers.add(node_name)
        for node in reversed(self.nodes):
            set_last_usage(node, future_used_buffers, self.mutation_real_name)
            future_used_buffers.update(node.last_usage)
        return fn(self, *args, **kwargs)
    return npu_sheduler_codegen


def npu_compute_ancestors(self) -> None:
    """
    Populate each node.ancestors
    """
    # note self.nodes is topologically sorted
    name_to_ancestors: Dict[str, OrderedSet[str]] = {}
    for node in self.nodes:
        ancestors: OrderedSet[str] = OrderedSet()
        for dep in node.unmet_dependencies:
            if dep.name not in self.name_to_buf:
                continue
            dep_node_name = self.name_to_buf[dep.name].defining_op.get_name()
            ancestors.add(dep_node_name)
            ancestors |= name_to_ancestors[dep_node_name]
        name_to_ancestors[node.get_name()] = ancestors
        node.ancestors = ancestors

    for order, node in enumerate(self.nodes):
        node.min_order = order
        node.max_order = order


def _npu_prune_redundant_deps(
    node: BaseSchedulerNode,
    name_to_fused_node: Dict[str, BaseSchedulerNode],
    name_to_buf: Dict[str, SchedulerBuffer],
) -> None:
    """
    Prunes weakdeps intended for mutation ordering
    on an upstream fused node if after fusion there is another dependency
    on the fused upstream node, making the weakdep redundant

    In essence this enforces an ordering on fusions. As fusions occur, weakdeps will
    be incrementally removed, enabling other fusions, ensuring they are fused in order.
    """
    name_to_dep_count: Counter[str] = collections.Counter()

    for dep in node.unmet_dependencies:
        if not isinstance(dep, WeakDep) and dep.name in name_to_buf:
            op = name_to_buf[dep.name].defining_op
            name_to_dep_count[name_to_fused_node[op.get_name()].get_name()] += 1

    def should_prune(dep: Dep) -> bool:
        if isinstance(dep, WeakDep) and dep.name in name_to_buf:
            op_name = name_to_buf[dep.name].defining_op.get_name()
            is_redundant = name_to_dep_count[name_to_fused_node[op_name].get_name()] > 0
            # These can occur because fused nodes always gather deps from their snodes
            # If B has a weakdep on A
            # B gets fused with C, then any time BC is fused, the weakdep will reappear
            is_self_dep = name_to_fused_node[op_name] == node
            return is_redundant or is_self_dep
        else:
            return False

    deps_to_prune = OrderedSet(
        dep for dep in node.unmet_dependencies if should_prune(dep)
    )

    if deps_to_prune:
        node.unmet_dependencies = node.unmet_dependencies - deps_to_prune
        node.set_read_writes(node.read_writes.remove_reads(deps_to_prune))


def npu_can_fuse_vertical(
    self,
    node1: BaseSchedulerNode,
    node2: BaseSchedulerNode,
    *,
    index_equivalent_dep_names: OrderedSet[str] | None = None,
) -> bool:
    """
    Check if it is legal to fuse a consumer (node2) into a producer (node1).

    We can fuse them if all the reads of node2 either match
    corresponding writes in node1, or are written by nodes that can
    be scheduled before the fusion of node1 and node2.

    ``index_equivalent_dep_names`` relaxes write/read matching only for
    named producer outputs; the remaining intermediate-dependency checks
    still run normally.
    """
    node1_buf_names = node1.get_buffer_names()
    node1_op_names = node1.get_operation_names()
    computed_deps: OrderedSet[Dep] = OrderedSet()
    why = WhyNoFuse(node1, node2)

    for cd in node1.read_writes.writes:
        if not isinstance(cd, MemoryDep):
            continue
        write_name = self.mutation_renames.get(cd.name, cd.name)
        fusable_kwargs = (
            {"allow_index_equivalence": write_name in index_equivalent_dep_names}
            if index_equivalent_dep_names is not None
            else {}
        )
        for rd in node2.unmet_dependencies:
            if self.fusable_read_and_write(rd, cd, **fusable_kwargs):
                computed_deps.add(rd)

    for dep in node2.unmet_dependencies:
        if isinstance(dep, WeakDep) and self.fusable_weak_dep(dep, node1, node2):
            computed_deps.add(dep)

    remaining_deps = OrderedSet(
        dep.name for dep in node2.unmet_dependencies - computed_deps
    )
    if remaining_deps & node1_buf_names:
        # MemoryDeps didn't match and read different locations of the same buffer.
        # Examples here include:
        #   - MemoryDep("foo", x) != MemoryDep("foo", x + 1)
        #   - MemoryDep("foo", x) != StarDep("foo")
        why("memory deps did not match")
        return False
    for name in remaining_deps:
        if name not in self.name_to_buf:
            continue
        op_name = self.name_to_buf[name].defining_op_name()
        if node1_op_names & self.name_to_fused_node[op_name].ancestors:
            why("intermediate nodes between node1 & node2")
            return False

    return True


def _npu_get_unmet_dep_nodes(self, snode: BaseSchedulerNode) -> List[BaseSchedulerNode]:
    unmet_deps = set()
    if isinstance(
        snode,
        (
            SchedulerNode,
            ExternKernelSchedulerNode,
            NopKernelSchedulerNode,
            FusedSchedulerNode,
        ),
    ):
        for dep in snode.unmet_dependencies:
            unmet_deps.add(dep.name)
    else:
        raise RuntimeError(
            f"get_unmet_dep_nodes is not implemented for {type(snode)}."
        )
    unmet_dep_ops = (self.name_to_buf[dep].defining_op for dep in unmet_deps if dep in self.name_to_buf)
    return list({self.name_to_fused_node[n.get_name()] for n in unmet_dep_ops})
