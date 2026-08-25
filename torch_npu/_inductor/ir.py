#!/usr/bin/env python3
from collections.abc import Callable, Sequence
from typing_extensions import Never
from typing import Optional

import sympy
from sympy import Expr, Integer

import torch
from torch._inductor import config
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.ir import log, Reduction, ReductionHint, Scatter, sympy_product
from torch._inductor.utils import ir_dataclass
from torch._inductor.virtualized import ops, V

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
            self.boundary,
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


def patch_triton_template_buffer_subgraph_symbols():
    """Preserve FlexAttention subgraph symbol dependencies on PyTorch 2.7.1.

    PyTorch 2.7.1's TritonTemplateBuffer does not expose the captured
    subgraph inputs and outputs to ``get_unbacked_symbol_uses``. NPU
    FlexAttention stores those dependencies on the selected template after
    autotuning; extend the class once so all template selection paths see
    them, including MultiTemplateBuffer.
    """
    from torch._inductor import ir
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    cls = ir.TritonTemplateBuffer
    if getattr(cls, "_npu_subgraph_symbol_patch", False):
        return

    original_init = cls.__init__
    original_get_uses = cls.get_unbacked_symbol_uses

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.subgraph_inps = None
        self.subgraph_outs = None

    def patched_get_unbacked_symbol_uses(self):
        uses = original_get_uses(self)

        for inp in getattr(self, "subgraph_inps", None) or ():
            if isinstance(inp, sympy.Expr):
                uses |= free_unbacked_symbols(inp)
            elif isinstance(inp, ir.IRNode):
                uses |= inp.get_unbacked_symbol_uses()
            elif inp is not None:
                raise TypeError(
                    "Unsupported FlexAttention subgraph input: "
                    f"{type(inp).__name__}"
                )

        for out in getattr(self, "subgraph_outs", None) or ():
            if isinstance(out, ir.IRNode):
                uses |= out.get_unbacked_symbol_uses()
            elif out is not None:
                raise TypeError(
                    "Unsupported FlexAttention subgraph output: "
                    f"{type(out).__name__}"
                )

        return uses

    cls.__init__ = patched_init
    cls.get_unbacked_symbol_uses = patched_get_unbacked_symbol_uses
    cls._npu_subgraph_symbol_patch = True
