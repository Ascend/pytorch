# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
import sympy

from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.ir import Layout
from torch._inductor.ops_handler import OpsHandler
from torch._inductor.shape_propagation import ShapePropagationOpsHandler
from torch._inductor.virtualized import V


def _has_negative_integer_power(expr):
    return isinstance(expr, sympy.Basic) and any(
        isinstance(node, sympy.Pow)
        and node.exp.is_integer
        and node.exp.is_negative
        for node in sympy.preorder_traversal(expr)
    )


def _install_safe_stride_order():
    marker = "_npu_safe_stride_order_installed"
    if getattr(Layout, marker, False):
        return

    def _safe_stride_expr_ge_or_false(left, right):
        sizevars = V.graph.sizevars
        if sizevars.guard_or_false(sympy.Eq(right, 0)):
            return True
        if sizevars.guard_or_false(sympy.Eq(left, 0)):
            return False
        if sizevars.guard_or_false(sympy.Ge(left, right)):
            return True

        divisible = sympy.Eq(left % right, 0)
        if _has_negative_integer_power(divisible):
            return False
        return sizevars.guard_or_false(divisible)

    Layout._stride_expr_ge_or_false = staticmethod(_safe_stride_expr_ge_or_false)
    setattr(Layout, marker, True)


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
