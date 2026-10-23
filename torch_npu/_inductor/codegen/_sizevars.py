import functools

import sympy
from sympy import Expr
from torch._inductor import ir, sizevars
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import free_symbols
from torch.utils._sympy.functions import Mod, safe_gcd, simple_floordiv_gcd


def simplify(self, expr: Expr):
    if isinstance(expr, (tuple, list)):
        return [sympy.expand(s).xreplace(self.replacements) for s in expr]
    return sympy.expand(expr).xreplace(self.replacements)


def _statically_known_stride_multiple_of(allocator, numerator, denominator):
    # The 2.13 allocator lacks the symbolic-denominator cancellation used by
    # upstream 93879b11e3b5. Keep that dependency local to stride proofs.
    if numerator is denominator or numerator == denominator:
        return True
    if isinstance(denominator, (int, sympy.Integer)):
        return allocator.statically_known_multiple_of(numerator, denominator)
    if numerator == 0:
        return True
    if len(free_symbols([numerator, denominator])) > sizevars._MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS:
        return False

    def gcd_covers_denominator(gcd):
        return gcd != 1 and (gcd == denominator or allocator.simplify(gcd - denominator) == 0)

    try:
        if gcd_covers_denominator(simple_floordiv_gcd(numerator, denominator)):
            return True
        if gcd_covers_denominator(safe_gcd(numerator, denominator)):
            return True
    except sympy.PolynomialError:
        pass
    return allocator.statically_known_true(sympy.Eq(Mod(numerator, denominator), 0))


def patch_simplify():
    sizevars.SizeVarAllocator.simplify = simplify
    original = getattr(ir.Layout, "_stride_expr_ge_or_false", None)
    if original is None or getattr(original, "_torch_npu_integer_strides", False):
        return

    @functools.wraps(original)
    def compare_strides(left, right):
        sizevars = V.graph.sizevars
        if sizevars.guard_or_false(sympy.Eq(right, 0)):
            return True
        if sizevars.guard_or_false(sympy.Eq(left, 0)):
            return False
        # Backport pytorch/pytorch@93879b11e3b5: prove unbacked divisibility
        # structurally, and avoid reciprocal powers in backed-symbol guards.
        return (
            sizevars.guard_or_false(sympy.Ge(left, right))
            or _statically_known_stride_multiple_of(sizevars, left, right)
            or sizevars.guard_or_false(sympy.Eq(Mod(left, right), 0))
        )

    compare_strides._torch_npu_integer_strides = True
    ir.Layout._stride_expr_ge_or_false = staticmethod(compare_strides)
