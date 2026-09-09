import torch
from torch._inductor import graph as inductor_graph

LazyString = inductor_graph.LazyString
OrderedSet = inductor_graph.OrderedSet
Pointwise = inductor_graph.Pointwise
Reduction = inductor_graph.Reduction
StorageBox = inductor_graph.StorageBox
TensorBox = inductor_graph.TensorBox
constrain_to_fake_tensors = inductor_graph.constrain_to_fake_tensors
constrain_to_fx_strides = inductor_graph.constrain_to_fx_strides
fallback_handler = inductor_graph.fallback_handler
fallback_node_due_to_unsupported_type = (
    inductor_graph.fallback_node_due_to_unsupported_type
)
gather_origins = inductor_graph.gather_origins
ir = inductor_graph.ir
log = inductor_graph.log
make_channels_last_strides_for = inductor_graph.make_channels_last_strides_for
needs_realized_inputs = inductor_graph.needs_realized_inputs
resolve_unbacked_bindings = inductor_graph.resolve_unbacked_bindings
GraphLowering = inductor_graph.GraphLowering


def patch_count_bytes():
    def count_bytes(self):
        total_bytes = 0
        node_counts = []
        node_runtimes = []
        for node in self.scheduler.nodes:
            try:
                num_bytes = node.get_read_write_buffers_sizes()
            except AssertionError:
                num_bytes = 0
            total_bytes += num_bytes
            node_counts.append((node, num_bytes // 4))
            node_runtimes.append((node, node.get_estimated_runtime()))

        return total_bytes, node_counts, node_runtimes

    torch._inductor.graph.GraphLowering.count_bytes = count_bytes
