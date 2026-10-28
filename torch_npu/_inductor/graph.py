from typing import Any

import torch
from torch._inductor import graph as inductor_graph
from torch.fx.node import Node

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


def _mutation_already_reflected(old_arg: object, new_arg: object) -> bool:
    if old_arg is new_arg:
        return True
    if isinstance(old_arg, ir.IRNode) and isinstance(new_arg, ir.IRNode):
        buffer_name = old_arg.maybe_get_name()
        return buffer_name is not None and buffer_name == new_arg.maybe_get_name()
    return False


def npu_propagate_mutation(
    self,
    fx_node: Node,
    old_args: tuple[Any],
    old_kwargs: dict[str, Any],
    new_args: tuple[Any],
    new_kwargs: dict[str, Any],
) -> None:
    """Backport upstream alias checks for layout-constraint mutation copy-back.

    These pairs represent the same argument before and after layout constraints,
    not arbitrary copy_ operands. If they alias the same buffer, the mutation is
    already visible and copy-back would risk creating a scheduler self-edge.
    Keep the 2.13 copy-back lowering for actual clones; newer mutate_to changes
    are intentionally not required by this backport.

    Upstream: pytorch/pytorch#189288, torch/_inductor/graph.py,
    GraphLowering.propagate_mutation (already_reflected).
    """
    assert len(old_args) == len(new_args)
    assert len(old_kwargs) == len(new_kwargs)

    if fx_node.target is torch.ops.higher_order.triton_kernel_wrapper_mutation:
        kwargs = fx_node.kwargs["kwargs"]
        assert isinstance(kwargs, dict)
        mutated = torch._higher_order_ops.triton_kernel_wrap.get_mutated_tensors(
            old_kwargs["kernel_idx"],
            old_kwargs["constant_args_idx"],
            {
                key: value.meta["val"] if isinstance(value, Node) else value
                for key, value in kwargs.items()
            },
            old_kwargs["tma_descriptor_metadata"],
        )
        for name in mutated:
            old_arg = old_kwargs["kwargs"][name]
            new_arg = new_kwargs["kwargs"][name]
            if _mutation_already_reflected(old_arg, new_arg):
                continue
            self.call_function(torch.ops.aten.copy_.default, (old_arg, new_arg), {})
        return

    assert isinstance(fx_node.target, torch._ops.OpOverload)

    def maybe_propagate(schema_arg, old_arg, new_arg):
        if old_arg is new_arg:
            return
        if schema_arg.alias_info is not None and schema_arg.alias_info.is_write:
            if isinstance(old_arg, ir.IRNode):
                old_arg = (old_arg,)
                new_arg = (new_arg,)
            for old_item, new_item in zip(old_arg, new_arg):
                if _mutation_already_reflected(old_item, new_item):
                    continue
                self.call_function(torch.ops.aten.copy_.default, (old_item, new_item), {})

    schema = fx_node.target._schema
    for index, (old_arg, new_arg) in enumerate(zip(old_args, new_args)):
        maybe_propagate(schema.arguments[index], old_arg, new_arg)

    schema_kwargs = {arg.name: arg for arg in schema.arguments}
    for key, old_arg in old_kwargs.items():
        maybe_propagate(schema_kwargs[key], old_arg, new_kwargs[key])


def patch_propagate_mutation():
    # Resolve the live class: switching another backend may reload graph.py.
    from torch._inductor.graph import GraphLowering

    GraphLowering.propagate_mutation = npu_propagate_mutation


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
