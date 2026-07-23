# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Custom IR kernels and ops-handler extensions for the triton_experimental backend.

``NPUCatLoopKernel`` is the universal single-kernel concat used by the NPU cat
lowering. The ``cat_store`` and ``index_select`` ops are registered on the inductor
ops handler (and its dtype/shape propagation handlers) here so that kernel and the
A5 gather lowering can emit them. Importing this module installs those handlers as a
side effect; ``lowering.py`` pulls ``NPUCatLoopKernel`` from here.
"""
import sympy

import torch
from torch.utils._sympy.functions import Identity
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.ir import Pointwise
from torch._inductor.ops_handler import OpsHandler
from torch._inductor.shape_propagation import ShapePropagationOpsHandler
from torch._inductor.utils import ir_dataclass
from torch._inductor.virtualized import ops, V


def _cat_store_op(self, output_name, store_index, value, mask):
    return self._default(
        "cat_store", (output_name, store_index, value, mask), {}
    )


OpsHandler.cat_store = _cat_store_op
# cat_store flows through the generic _default path (not the store fast-path) and
# returns its value operand unchanged, so dtype/shape mirror value, not promotion.


def _cat_store_dtype(output_name, store_index, value, mask):
    return value.dtype


def _cat_store_shape(output_name, store_index, value, mask):
    return getattr(value, "shape", None)


DtypePropagationOpsHandler.cat_store = staticmethod(_cat_store_dtype)
ShapePropagationOpsHandler.cat_store = staticmethod(_cat_store_shape)


# A5 CANN row gather: ops.index_select(src_name, weight_index, indirect_var,
# set_indirect, bound) codegens to __builtin_index_select. Like cat_store it flows
# through _default, so dtype/shape mirror the returned tile (weight dtype).
def _index_select_op(self, src_name, weight_index, indirect_var, set_indirect, bound):
    return self._default(
        "index_select",
        (src_name, weight_index, indirect_var, set_indirect, bound),
        {},
    )


OpsHandler.index_select = _index_select_op


def _index_select_dtype(src_name, weight_index, indirect_var, set_indirect, bound):
    return V.graph.get_dtype(src_name)


def _index_select_shape(src_name, weight_index, indirect_var, set_indirect, bound):
    return getattr(indirect_var, "shape", None)


DtypePropagationOpsHandler.index_select = staticmethod(_index_select_dtype)
ShapePropagationOpsHandler.index_select = staticmethod(_index_select_shape)


@ir_dataclass
class NPUCatLoopKernel(Pointwise):
    """Universal single-kernel concat (any axis, any shape, dynamic).

    A shared-tile store with local index ``x0 - lo`` runs the contiguous DMA base
    NEGATIVE on an input's foreign lanes (MTE 507035 fault, even masked). Instead:
    load at the 0-based local coord ``x0`` (base ``row*size_i + x0`` always >= 0;
    foreign lanes forward-overshoot and are masked), store at ``lo + x0``, mask
    ``x0 < size_i``. Bit-exact for first/middle/tail axes, size-1 lanes, dynamic.
    """

    cat_inputs: tuple = ()
    cat_dim: int = 0
    cat_ranges: tuple = ()

    def store_output(self, output_name, indexer, vars):
        c = vars[self.cat_dim]
        idx_c = ops.index_expr(c, torch.int64)
        for i, inp in enumerate(self.cat_inputs):
            lo, hi = self.cat_ranges[i]
            size_i = inp.get_size()[self.cat_dim]

            # Load at 0-based local coord c (NOT c - lo): base stays non-negative.
            idx_load = list(vars)
            idx_load[self.cat_dim] = Identity(c)

            def load(il=idx_load, i=i):
                return self.cat_inputs[i].make_loader()(il)

            end = ops.index_expr(size_i, torch.int64)
            mask = ops.lt(idx_c, end)
            val = ops.masked(mask, load, 0.0)

            idx_store = list(vars)
            idx_store[self.cat_dim] = Identity(sympy.expand(lo + c))
            ops.cat_store(output_name, indexer(idx_store), val, mask)
        return None
