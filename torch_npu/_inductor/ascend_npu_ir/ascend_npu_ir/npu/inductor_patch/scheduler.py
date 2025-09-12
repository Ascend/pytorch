import torch
import sympy
import collections
from typing import (
    Union,
    Optional
)

from torch._inductor import ir
from torch._inductor import scheduler
from torch._inductor.scheduler import (
    SchedulerNode,
    BaseSchedulerNode,
    FusedSchedulerNode,
    NopKernelSchedulerNode,
    ExternKernelSchedulerNode,
    OutputNode,
    MultiOutput,
    MultiOutputLayout,
    OrderedSet,
    Sequence,
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
        return V.graph.sizevars.size_hint(s, fallback=0)

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