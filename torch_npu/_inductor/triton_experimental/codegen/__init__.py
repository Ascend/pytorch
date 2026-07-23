# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Register LoopBody index-capture handlers for the custom ``cat_store`` op (used
# by NPUCatLoopKernel). Its sympy index args must go through the LoopBody index
# machinery, not the FX tracer raw. CaptureIndexing registers during tracing;
# SimplifyIndexing simplifies at replay. The op otherwise flows through
# OpsHandler.cat_store (lowering.py) and NPUTritonKernelOverrides.cat_store.
from torch._inductor.loop_body import CaptureIndexing, MemoryUsageType
from torch._inductor.sizevars import SimplifyIndexing


def _loop_body_block_cat_store(self, output_name, store_index, value, mask):
    # store_index is the only sympy arg; value/mask are ops results (concat coord
    # and bounds are already folded into mask at lowering time).
    store_index = self._simplify(store_index)
    store_index = self._add_index(
        store_index, MemoryUsageType.STORE, buffer_name=output_name
    )
    return self._inner.cat_store(output_name, store_index, value, mask)


def _simplify_indexing_cat_store(self, output_name, store_index, value, mask):
    return self._inner.cat_store(
        output_name,
        self._simplify(store_index),
        value,
        mask,
    )


CaptureIndexing.cat_store = _loop_body_block_cat_store
SimplifyIndexing.cat_store = _simplify_indexing_cat_store


# ``index_select`` (A5 __builtin_index_select gather) carries one sympy arg,
# ``weight_index`` (flat gather index into the weight, embedding the indirect TMP
# row symbol). Like an indirect load it registers via MemoryUsageType.LOAD; the
# row-index var, indirect symbol name, and bound thread through unchanged.
def _loop_body_block_index_select(self, src_name, weight_index, indirect_var, set_indirect, bound):
    weight_index = self._simplify(weight_index)
    weight_index = self._add_index(
        weight_index, MemoryUsageType.LOAD, buffer_name=src_name
    )
    return self._inner.index_select(src_name, weight_index, indirect_var, set_indirect, bound)


def _simplify_indexing_index_select(self, src_name, weight_index, indirect_var, set_indirect, bound):
    return self._inner.index_select(
        src_name,
        self._simplify(weight_index),
        indirect_var,
        set_indirect,
        bound,
    )


CaptureIndexing.index_select = _loop_body_block_index_select
SimplifyIndexing.index_select = _simplify_indexing_index_select
