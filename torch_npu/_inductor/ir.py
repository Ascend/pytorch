#!/usr/bin/env python3
from collections.abc import Callable, Sequence
from typing_extensions import Never

from sympy import Expr, Integer

import torch
from torch._inductor import config
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.ir import log, Reduction, ReductionHint, Scatter, sympy_product
from torch._inductor.utils import ir_dataclass
from torch._inductor.virtualized import ops, V

@ir_dataclass
class IndexputTemplate(Scatter):
    boundary: int | None = None

    def store_output(
        self,
        output_name: str | None,
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
            self.boundary,
        )


class ScatterTemplate(Scatter):
    def store_output(
        self,
        output_name: str | None,
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
    def get_hint(x):
        if isinstance(x, (int, float)):
            return x
        try:
            return int(V.graph.sizevars.size_hint(x))
        except Exception:
            return 1

    ranges = [h for num in reduction_ranges if (h := get_hint(num)) > 1]
    if not ranges:
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
