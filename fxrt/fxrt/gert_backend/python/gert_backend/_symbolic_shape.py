"""Adapt PyTorch layout expressions to the existing FXRT symbolic algebra."""

from functools import reduce

import sympy
import torch

from fxrt.ir import SymbolicConst
from fxrt.symbolic_shape import SymbolicShapeManager


class ConverterSymbolicShapeManager(SymbolicShapeManager):
    """Keep Max expressions exact without sampling sizes or adding guards."""

    def export_int_list(self, values):
        """Serialize layout expressions through the converter's existing IR ABI."""
        elements = []
        for value in values:
            if isinstance(value, torch.SymInt) and not value.node.expr.is_Integer:
                expression = self.convert_sympy_expr_to_symbolic_expr(value.node.expr)
                elements.append({"kind": "symbol", "value": repr(expression)})
            else:
                literal = int(value.node.expr) if isinstance(value, torch.SymInt) else int(value)
                elements.append({"kind": "int", "value": literal})
        return {"kind": "tuple", "elements": elements}

    def convert_sympy_expr_to_symbolic_expr(self, expr):
        if expr.func is sympy.Max or getattr(expr.func, "__name__", "") == "Max":
            # FXRT already represents Min and signed integer arithmetic:
            # max(a, b) = -min(-a, -b). Reuse those operations in both runners.
            negative = SymbolicConst(-1)
            values = [negative * self.convert_sympy_expr_to_symbolic_expr(arg) for arg in expr.args]
            return negative * reduce(lambda lhs, rhs: lhs.__min__(rhs), values)
        return super().convert_sympy_expr_to_symbolic_expr(expr)
