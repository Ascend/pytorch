import itertools
import operator
from typing import Any

import sympy

import torch
from torch._dynamo.utils import defake
from torch._inductor import config, graph as inductor_graph, metrics
from torch._inductor.utils import clone_preserve_strides
from torch._inductor.virtualized import NullHandler, V
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.fx.node import Node
from torch.utils._sympy.numbers import int_oo


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
is_magic_method = inductor_graph.is_magic_method
log = inductor_graph.log
make_channels_last_strides_for = inductor_graph.make_channels_last_strides_for
needs_realized_inputs = inductor_graph.needs_realized_inputs
resolve_unbacked_bindings = inductor_graph.resolve_unbacked_bindings
GraphLowering = inductor_graph.GraphLowering


def patch_codegen_with_cpp_wrapper():
    """
    patch codegen for cpp wrapper, add npu for codegen_with_cpp_wrapper function

    """
    def npu_codegen_with_cpp_wrapper(self) -> tuple[str, list[tuple[int, Node]]]:
        # add "npu" support
        if any(device in self.device_types for device in ["cuda", "xpu", "npu"]):
            if config.triton.autotune_at_compile_time:
                # If autotune_at_compile_time is True, we can do the codegen in one-pass
                return self.codegen()
            else:
                # first pass
                self.cpp_wrapper = False
                compiled = self.compile_to_module().call

                def materialize(
                    x: torch.SymInt | torch.SymFloat | torch.Tensor,
                ) -> int | float | torch.Tensor:
                    if x is None:
                        return None
                    elif isinstance(x, (torch.SymInt, torch.SymFloat)):
                        # Need concrete value to run dynamic shapes and tune the result
                        return x.node.hint
                    elif isinstance(x, FakeTensor):
                        return defake(x)
                    else:
                        if not isinstance(x, torch.Tensor):
                            raise AssertionError(
                                "Unknown type when creating real inputs" + str(type(x))
                            )
                        return x

                tracing_context = torch._guards.TracingContext.try_get()
                if tracing_context is not None and not isinstance(
                    V.real_inputs, NullHandler
                ):
                    if tracing_context.output_strides:
                        tracing_context.output_strides.clear()

                    params_flat = [
                        param
                        for param in tracing_context.params_flat  # type: ignore[union-attr]
                        if param is not None
                    ]
                    real_inputs = [
                        materialize(x)
                        for x in itertools.chain(params_flat, V.real_inputs)
                    ]
                else:
                    # In the backward pass, V.real_inputs is not OrderedSet.
                    # Generating random inputs based on self.example_inputs sometimes can be problematic,
                    # e.g. illegal memory access. A comprehensive fix is to autotune in a separate process.
                    real_inputs = [
                        materialize(x)  # type:ignore[arg-type]
                        for x in (
                            self.example_inputs  # type:ignore[union-attr]
                            if isinstance(V.real_inputs, NullHandler)
                            else V.real_inputs
                        )
                    ]

                if self.mutated_inputs:
                    mutated_input_idxs = [
                        idx
                        for idx, name in enumerate(self.graph_inputs)
                        if name in self.mutated_inputs
                        and isinstance(real_inputs[idx], torch.Tensor)
                    ]
                    for idx in mutated_input_idxs:
                        # clone mutated Tensor inputs to avoid mutating them in
                        # the first pass of the CPP wrapper-based compilation, as
                        # this will lead to a side effect on the example inputs:
                        # e.g. if torch.compile(f)(x) if called on input-mutating
                        # f, the inputs x will be mutated twice in the process:
                        # once here, and again when running the compiled model;
                        # this will also lead to a numerically incorrect output
                        mutated_inp = real_inputs[idx]
                        if not isinstance(mutated_inp, torch.Tensor):
                            raise AssertionError
                        real_inputs[idx] = clone_preserve_strides(mutated_inp)
                        del mutated_inp

                with torch.utils._python_dispatch._disable_current_modes():
                    compiled(real_inputs)
                del real_inputs

                # second pass
                self.cpp_wrapper = True
                self.removed_buffers.clear()
                self.removed_operations.clear()
                self.inplaced_to_remove.clear()
                V.graph.sizevars.precomputed_replacements.clear()
                V.graph.sizevars.inv_precomputed_replacements.clear()
                metrics.reset()
                with config.patch({"triton.autotune_at_compile_time": False}):
                    return self.codegen()
        else:
            # cpu
            return self.codegen()

    from torch._inductor.graph import GraphLowering

    GraphLowering.codegen_with_cpp_wrapper = npu_codegen_with_cpp_wrapper


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
