import sympy
from sympy import Expr
from torch._inductor.utils import sympy_subs


def simplify(self, expr: Expr):
    if isinstance(expr, (tuple, list)):
        return [sympy.expand(s).xreplace(self.replacements) for s in expr]
    return sympy.expand(expr).xreplace(self.replacements)
