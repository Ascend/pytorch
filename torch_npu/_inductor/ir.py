from typing import (
    Callable,
    Optional,
)
from typing_extensions import Never
from collections.abc import Sequence

from sympy import Expr, Integer
import torch
from torch._inductor.ir import (NopKernel, SliceView, IRNode, StorageBox,
                                FlexibleLayout, FixedLayout, NonOwningLayout,
                                Pointwise, Reduction, log, Scatter,
                                ReductionHint, IRNode, sympy_product)
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.virtualized import ops, V
from torch._inductor.utils import ir_dataclass
from torch._inductor import config
from torch.utils._sympy.functions import Identity

from . import config as npu_config


def patch_fallback_kernel_codegen():
    from torch._inductor.ir import FallbackKernel
    origin_fallback_codegen = FallbackKernel.codegen

    # todo:
    # 1. merge fallback_ops in fallback_ops.py and lowering_fallback_list.py
    # 2. let torchnpugen support update-aoti-c-shim
    # 3. register external kernel to c-shim kernel
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
        origin_fallback_codegen(self, wrapper)

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

            max_numel_in_per_kernel = npu_config.max_cat_size_in_per_kernel // get_byte_per_numel(inputs[0].get_dtype())
            input_sub = []
            prev = 0
            for i, inp in enumerate(inputs):
                input_sub.append(inp)
                cat_count_overflow = npu_config.max_cat_count_in_per_kernel is not None and (i - prev + 1 >= npu_config.max_cat_count_in_per_kernel)
                if i == len(inputs) - 1 or offsets_end[i + 1] - offsets_start[prev] > max_numel_in_per_kernel or cat_count_overflow:
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
        if npu_config.use_store_in_cat or is_split_inputs:
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


def reduction_split_factor(reduction_ranges):
    ranges = [num for num in reduction_ranges if num > 1]
    if len(ranges) == 0:
        return 1
    return min(ranges)


def num_splits(
    device,
    dst_dtype,
    src_dtype,
    inner_fn,
    ranges,
    reduction_ranges,
    reduction_type,
    reduction_numel,
    input_node=None,
):
    def _is_static(x: object) -> bool:
        return isinstance(x, (int, Integer))

    reduction_numel_hint = V.graph.sizevars.symbolic_hint(reduction_numel)
    numel_hint = V.graph.sizevars.symbolic_hint(sympy_product(ranges))
    if not (_is_static(reduction_numel_hint) and _is_static(numel_hint)):
        # We don't support unbacked symints
        return ReductionHint.DEFAULT, 1

    should_split = reduction_type == "scan" or (
        not V.graph.has_feature(device, BackendFeature.REDUCE_TO_SINGLE_ELEMENT)
        and reduction_type
        not in (
            "argmax",
            "argmin",
        )
        and config.split_reductions
    )

    if should_split:
        inner_reduction_splits = reduction_split_factor
    else:
        def inner_reduction_splits(reduction_ranges):
            return 1

    if numel_hint == 1:
        split = inner_reduction_splits(reduction_ranges)
        return ReductionHint.INNER, split
    return ReductionHint.DEFAULT, 1

def patch_num_splits():
    Reduction.num_splits = num_splits
