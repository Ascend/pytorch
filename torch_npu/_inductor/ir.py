from typing import (
    Callable,
    Optional,
)
from typing_extensions import Never
from collections.abc import Sequence

from sympy import Expr, Integer
import torch
from torch._inductor.ir import Reduction, log, Scatter, ReductionHint, sympy_product
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.virtualized import ops, V
from torch._inductor.utils import ir_dataclass
from torch._inductor import config


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
