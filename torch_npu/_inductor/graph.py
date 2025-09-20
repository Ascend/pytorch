from typing import (
    Any,
    List,
    Tuple,
    Union,
)
import itertools

import torch
from torch.fx.node import Node
from torch._inductor import config, metrics
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo.utils import defake, dynamo_timed
from torch._inductor.virtualized import NullHandler, V


def patch_codegen_with_cpp_wrapper():
    def npu_codegen_with_cpp_wrapper(self) -> Tuple[str, List[Tuple[int, Node]]]:
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
                    x: Union[torch.SymInt, torch.SymFloat, torch.Tensor]
                ) -> Union[int, float, torch.Tensor]:
                    if x is None:
                        return None
                    elif isinstance(x, (torch.SymInt, torch.SymFloat)):
                        # Need concrete value to run dynamic shapes and tune the result
                        return x.node.hint
                    elif isinstance(x, FakeTensor):
                        return defake(x)
                    else:
                        if not isinstance(x, torch.Tensor):
                            raise AssertionError("Unknown type when creating real inputs" + str(type(x)))
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
                    from .compile_fx import clone_preserve_strides

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