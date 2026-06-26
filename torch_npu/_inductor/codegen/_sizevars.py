import sympy
from sympy import Expr
from torch._inductor import sizevars


def simplify(self, expr: Expr):
    if isinstance(expr, (tuple, list)):
        return [sympy.expand(s).xreplace(self.replacements) for s in expr]
    return sympy.expand(expr).xreplace(self.replacements)

def patch_simplify():
    sizevars.SizeVarAllocator.simplify = simplify