"""
Symbolic shape management for handling dynamic tensor dimensions.
"""
import operator
from functools import reduce
from typing import List, Dict, Any, Optional

import torch
import sympy

from torch.utils._sympy.functions import (
    FloorDiv,
    IntTrueDiv,
    CeilDiv,
    FloorToInt,
    CeilToInt,
)
from fxrt.ir import SymbolicVar, SymbolicConst, SymbolicExpr, Value, Tuple
from fxrt.utils import from_torch


class SymbolicShapeManager:
    """
    Manages symbolic shape creation and conversion for tensors with dynamic dimensions.

    This class handles the conversion between sympy expressions and FXRT symbolic expressions,
    and recursively processes nested tensor structures (lists, tuples) to create symbolic shapes.
    """

    _SYMPY_EXPR_TO_SYMBOLIC_OP = {
        sympy.Add: operator.add,
        sympy.Mul: operator.mul,
        sympy.Mod: operator.mod,
        FloorDiv: operator.floordiv,
        CeilDiv: lambda a, b: a.__ceildiv__(b),
    }

    def __init__(self):
        # Map from symbol name (str) to SymbolicVar
        self._symbol_map: Dict[str, SymbolicVar] = {}

    def convert_sympy_expr_to_symbolic_expr(self, expr: sympy.Expr) -> SymbolicExpr:
        """
        Recursively convert a sympy expression to a SymbolicExpr.

        Args:
            expr: The sympy expression to convert

        Returns:
            The corresponding SymbolicExpr

        Raises:
            NotImplementedError: If the expression type is not supported
        """
        if expr.is_Symbol:
            sym_name = str(expr)
            if sym_name not in self._symbol_map:
                self._symbol_map[sym_name] = SymbolicVar(sym_name)
            return self._symbol_map[sym_name]

        if expr.is_Integer:
            return SymbolicConst(int(expr))

        if expr.is_Pow:
            base, exp = expr.as_base_exp()
            if not exp.is_Integer:
                raise NotImplementedError(
                    f"Unsupported sympy expression: {expr} (type: {type(expr)})"
                )
            expr = sympy.Mul(*[base] * exp, evaluate=False)

        # Eliminate ops with float outputs
        if isinstance(expr, FloorToInt) and isinstance(expr.args[0], IntTrueDiv):
            return self.convert_sympy_expr_to_symbolic_expr(
                FloorDiv(*expr.args[0].args)
            )

        if isinstance(expr, CeilToInt) and isinstance(expr.args[0], IntTrueDiv):
            return self.convert_sympy_expr_to_symbolic_expr(CeilDiv(*expr.args[0].args))

        # Handle Mod explicitly to avoid exact-type lookup misses.
        # Some runtimes may construct a Mod class object that is not `sympy.Mod`
        # by identity, so also match by function/class name.
        if (getattr(expr, "func", None) is sympy.Mod or getattr(getattr(expr, "func", None), "__name__",
                                                                None) == "Mod" or type(expr).__name__ == "Mod"):
            if len(expr.args) != 2:
                raise NotImplementedError(f"Unsupported Mod args: {expr} (args: {expr.args})")
            lhs, rhs = expr.args
            return (self.convert_sympy_expr_to_symbolic_expr(lhs) % self.convert_sympy_expr_to_symbolic_expr(rhs))

        # Handle Min explicitly to avoid exact-type lookup misses across sympy runtimes.
        if (getattr(expr, "func", None) is sympy.Min or getattr(getattr(expr, "func", None), "__name__",
                                                                None) == "Min" or type(expr).__name__ == "Min"):
            converted_args = [self.convert_sympy_expr_to_symbolic_expr(arg) for arg in expr.args]
            return reduce(lambda a, b: a.__min__(b), converted_args)

        # Convert basic symbolic ops
        op = self._SYMPY_EXPR_TO_SYMBOLIC_OP.get(type(expr))
        if op:
            converted_args = [
                self.convert_sympy_expr_to_symbolic_expr(arg) for arg in expr.args
            ]
            if isinstance(expr, (sympy.Add, sympy.Mul)):
                return reduce(op, converted_args)
            return op(*converted_args)
        raise NotImplementedError(
            f"Unsupported sympy expression: {expr} (type: {type(expr)})"
        )

    def create_symbolic_shape_for_tensor(self, tensor) -> Optional[List[SymbolicExpr]]:
        """
        Create symbolic shape for a tensor if it contains symbolic dimensions.

        Args:
            tensor: The torch tensor that may contain symbolic dimensions

        Returns:
            List of symbolic expressions for the tensor shape, or None if no symbolic dimensions
        """
        if not isinstance(tensor, torch.Tensor):
            return None

        if not any(isinstance(d, torch.SymInt) for d in tensor.shape):
            return None

        symbolic_shape = []
        for dim in tensor.shape:
            if isinstance(dim, torch.SymInt):
                expr = dim.node.expr
                symbolic_shape.append(self.convert_sympy_expr_to_symbolic_expr(expr))
            else:
                symbolic_shape.append(SymbolicConst(int(dim)))

        return symbolic_shape

    def from_torch_with_sym(self, torch_value: Any) -> Value:
        """
        Convert a torch object to fxrt.ir.Value with symbolic shape binding.

        Args:
            torch_value: The value from torch (can be Tensor, list, tuple, or nested structures)

        Returns:
            A new FXRT Value with symbolic shape information bound
        """
        # Handle SymInt case first - we need to create/retrieve SymbolicVar
        if isinstance(torch_value, torch.SymInt):
            expr = torch_value.node.expr
            return Value(self.convert_sympy_expr_to_symbolic_expr(expr))

        # Handle tensor case
        if isinstance(torch_value, torch.Tensor):
            # Convert tensor using from_torch
            fxrt_value = from_torch(torch_value)
            # Apply symbolic shape if needed
            symbolic_shape = self.create_symbolic_shape_for_tensor(torch_value)
            if symbolic_shape is not None:
                fxrt_value.to_tensor().symbolic_shape = symbolic_shape
            return fxrt_value

        # Handle nested lists/tuples recursively
        if isinstance(torch_value, (list, tuple)):
            # Convert each element recursively
            elements = [self.from_torch_with_sym(item) for item in torch_value]
            return Value(Tuple(elements))

        # For other types, just use from_torch
        return from_torch(torch_value)
