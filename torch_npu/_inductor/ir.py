from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    overload,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

from collections.abc import Sequence
import torch
from torch._inductor.ir import (NopKernel, SliceView, IRNode, StorageBox, FlexibleLayout, FixedLayout, NonOwningLayout, Pointwise, TensorBox, ComputedBuffer, View, log, Layout, Scatter)
from torch._inductor.virtualized import ops, OpsValue, V
from torch._inductor.utils import ir_dataclass
from torch._inductor import lowering
from sympy import Expr, Integer, Symbol
from typing_extensions import Never

from torch.utils._sympy.functions import Identity
from torch_npu._inductor import config


def patch_fallback_kernel_codegen():
    def codegen_npu(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        kernel = self.op_overload
        if kernel.namespace == "aten":  # type: ignore[union-attr]
            if not isinstance(kernel, torch._ops.OpOverload):
                raise AssertionError(f"kernel should be OpOverload, but got {type(kernel)}")
            if V.graph.cpp_wrapper:
                # Fallback all npu op to proxy executor and warn when gpu do not.
                from torchgen.aoti.fallback_ops import inductor_fallback_ops
                self.use_runtime_dispatch = True
                if str(kernel) in inductor_fallback_ops:
                    log.warning(
                        "%s is using proxy executor as fallback instead of aoti shim.",
                        kernel,
                    )

        elif kernel.namespace == "_quantized":  # type: ignore[union-attr]
            # Internal Quantized Fallback Ops
            if not isinstance(kernel, torch._ops.OpOverload):
                raise AssertionError
        else:
            # For non-aten OpOverload, i.e. custom ops
            if V.graph.cpp_wrapper:
                self.use_runtime_dispatch = True

        if self.use_runtime_dispatch:
            self.codegen_comment(wrapper)

            exported_args = None
            args = None
            exported_args = self.export_extern_kernel_node()

            wrapper.generate_fallback_kernel_with_runtime_lookup(
                self.get_name(),
                self.python_kernel_name,
                self.cpp_kernel_name,
                args,
                self.op_overload,
                exported_args,
                # NOTE: [special handling of all_reduce_coalesced_'s return value]
                self.outputs if self.outputs else self.mutation_outputs,
            )
        else:
            self.codegen_comment(wrapper)
            args = [*self.codegen_args(), *self.codegen_kwargs()]
            V.graph.wrapper_code.generate_fallback_kernel(self, args)
            if isinstance(self.layout, Layout):
                self.codegen_size_asserts(wrapper)

        self.codegen_unbacked_symbol_defs(wrapper)
    
    from torch._inductor.ir import FallbackKernel
    FallbackKernel.codegen = codegen_npu



class ConcatKernel(NopKernel):
    @classmethod
    def create(cls, inputs: Sequence[IRNode], dim: int, is_reindex: bool) -> StorageBox:
        new_size = list(inputs[0].get_size())
        offsets_start = [0]
        offsets_end = [new_size[dim]]

        for i in range(1, len(inputs)):
            input_size = inputs[i].get_size()
            offsets_start.append(new_size[dim])
            new_size[dim] = new_size[dim] + input_size[dim]
            offsets_end.append(new_size[dim])

        output_stride: Sequence[int] = FlexibleLayout.contiguous_strides(new_size)

        concat_kernel = ConcatKernel(
            name=None,
            layout=FixedLayout(
                device=inputs[0].get_device(),
                dtype=inputs[0].get_dtype(),
                size=new_size,
                stride=output_stride,
            ),
            inputs=[],
        )
        kernel = StorageBox(concat_kernel)

        if is_reindex:
            for i, inp in enumerate(inputs):
                input_buffer = cls.single_realize_into(inp, SliceView.create(
                    kernel, dim, offsets_start[i], offsets_end[i], clamp=False))
                concat_kernel.inputs.append(input_buffer)
        else:
            from torch_npu._inductor.codegen.triton_utils import get_byte_per_numel

            max_numel_in_per_kernel = config.max_cat_size_in_per_kernel // get_byte_per_numel(inputs[0].get_dtype())
            input_sub = []
            prev = 0
            for i, inp in enumerate(inputs):
                input_sub.append(inp)
                if i == len(inputs) - 1 or offsets_end[i + 1] - offsets_start[prev] > max_numel_in_per_kernel:
                    input_buffer = cls.realize_into(input_sub, SliceView.create(
                        kernel, dim, offsets_start[prev], offsets_end[i], clamp=False
                    ), dim)
                    concat_kernel.inputs.append(input_buffer)
                    input_sub = []
                    prev = i + 1

        concat_kernel.name = V.graph.register_buffer(concat_kernel)
        concat_kernel.inputs = cls.unwrap_storage(concat_kernel.inputs)
        V.graph.register_operation(concat_kernel)

        return kernel

    @classmethod
    def single_realize_into(cls, src: IRNode, dst: IRNode) -> IRNode:
        pw = Pointwise.create(
            device=src.get_device(),
            dtype=src.get_dtype(),
            inner_fn=src.make_loader(),
            ranges=src.get_size(),
        )
        pw.realize()
        pw.data.data.layout = NonOwningLayout(dst)
        return pw.data.data

    @classmethod
    def realize_into(cls, inputs: Sequence[IRNode], dst: IRNode, dim) -> IRNode:
        if len(inputs) == 1:
            return cls.single_realize_into(inputs[0], dst)

        inputs_ranges = [0]
        prev_end = 0
        for inp in inputs:
            inputs_ranges.append((prev_end + inp.get_size()[dim]))
            prev_end = inputs_ranges[-1]

        output_size = list(inputs[0].get_size())
        output_size[dim] = inputs_ranges[-1]

        def inner_fn_insert_slice(idx):
            idx_load = list(idx)
            output = ops.index_expr(output_size[dim], torch.float32)
            for i, inp in enumerate(inputs):
                output = ops.cat_insert_slice(output, inp.make_loader()(idx_load), int(inputs_ranges[i]),
                                 int(inp.get_size()[dim]), int(output_size[dim]))
            return output

        def inner_fn_store(idx):
            idx_load = list(idx)
            output = ops.index_expr(output_size[dim], torch.float32)
            for i, inp in enumerate(inputs):
                idx_output = list(idx)
                idx_output[dim] = Identity(idx_output[dim] + inputs_ranges[i])
                output = ops.cat_store(dst.get_name(), inp.make_loader()(idx_load), int(inp.get_size()[dim]),
                                 dst.make_indexer()(idx_output), dst.make_indexer()(idx_load))
            return output

        input_strides = [inp.get_stride()[dim - 1] == output_size[dim] for inp in inputs if inp.maybe_get_stride() is not None]
        is_split_inputs = input_strides and all(input_strides)
        if config.use_store_in_cat or is_split_inputs:
            pw = ConcatOutputKernel.create(
                device=inputs[0].get_device(),
                dtype=inputs[0].get_dtype(),
                inner_fn=inner_fn_store,
                ranges=output_size,
            )
        else:
            pw = Pointwise.create(
                device=inputs[0].get_device(),
                dtype=inputs[0].get_dtype(),
                inner_fn=inner_fn_insert_slice,
                ranges=output_size,
            )

        pw.realize()
        pw.data.data.layout = NonOwningLayout(dst)
        return pw.data.data


@ir_dataclass
class ConcatOutputKernel(Pointwise):
    def store_output(self, output_name, indexer, store_vars) -> None:
        loader = self.make_loader()
        loader(store_vars)
        return None


@ir_dataclass
class IndexputTemplate(Scatter):
    boundary: Optional[int] = None

    def store_output(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[Expr]], Never],
        store_vars: Sequence[Expr],
    ) -> None:
        loader = self.make_loader()
        if output_name is None:
            output_name = "unnamed"
        output_indexer = self.output_indexer(store_vars)
        indirect_indexer = None
        for var in output_indexer:
            if str(var).startswith("indirect"):
                indirect_indexer = var
                break

        return ops.indexput_template(
            output_name,
            indexer(output_indexer),
            loader(store_vars),
            indirect_indexer,
            self.boundary
        )


class ScatterTemplate(Scatter):
    def store_output(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[Expr]], Never],
        store_vars: Sequence[Expr],
    ) -> None:
        loader = self.make_loader()
        if output_name is None:
            output_name = "unnamed"
        output_indexer, boundary = self.output_indexer(store_vars)
        indirect_indexer = None
        for var in output_indexer:
            if str(var).startswith("indirect"):
                indirect_indexer = var
                break

        return ops.scatter_template(
            output_name,
            indexer(output_indexer),
            loader(store_vars),
            indirect_indexer,
            int(boundary),
        )