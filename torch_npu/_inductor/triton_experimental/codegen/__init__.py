# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
from torch._inductor.loop_body import CaptureIndexing, MemoryUsageType
from torch._inductor.sizevars import SimplifyIndexing

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
