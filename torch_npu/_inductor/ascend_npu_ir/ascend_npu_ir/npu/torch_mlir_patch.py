import math
import sympy
from typing import Optional, Dict, Union

import torch
from torch_mlir import ir
from torch_mlir.extras import fx_importer
from torch_mlir.compiler_utils import (
    OutputType
)
from torch_mlir.dialects import torch as torch_d
from torch_mlir.fx import (
    _module_lowering,
    FxImporter,
    FxImporterHooks,
)
from torch_mlir.extras import fx_importer

from torch_mlir.extras.fx_importer import (
    Graph,
    Operation,
    Callable,
    func_dialect,
    RangeConstraint,
    Block,
    GraphNodeImporter,
    UnitAttr,
    sympy_expr_to_semi_affine_expr
)

from torch_mlir.ir import (
    AffineAddExpr,
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineMapAttr,
    AffineModExpr,
    AffineMulExpr,
    AffineSymbolExpr,
    AffineFloorDivExpr
)

from torch.utils._sympy.functions import (
    CeilDiv,
    FloorDiv,
    Identity,
    IntTrueDiv,
    ModularIndexing,
)

def _patch_import_stateless_graph(
    self,
    g: Graph,
    *,
    func_name: str = "main",
    func_visibility: Optional[str] = None,
    import_symbolic_shape_expressions: bool = True,
) -> Operation:
    """Low-level import of a functionalized, assumed stateless Graph as a func.

    TODO: This mechanism is deprecated by the `import_program` entry-point and
    it should be removed when no longer required for backwards compatibility.
    """

    def get_range_constraints(graph: torch.fx.Graph):
        range_constraints = {}
        for nd in graph.find_nodes(
            op="placeholder"
        ):  
            if isinstance(nd.meta['val'], torch.Tensor):
                for s in nd.meta['val'].size():
                    if isinstance(s, torch.SymInt):
                        for symbol in s._sympy_().free_symbols:
                            range_constraints[symbol] = torch.utils._sympy.value_ranges.ValueRanges(128, 1024)
            else:
                for symbol in nd.meta['val']._sympy_().free_symbols:
                    range_constraints[symbol] = torch.utils._sympy.value_ranges.ValueRanges(128, 1024)
        return range_constraints


    def _sympy_int_to_int(val: sympy.Expr, adjust_func: Callable):
        # Convert simple sympy Integers into concrete int
        if val == sympy.oo:
            return math.inf
        if val == -sympy.oo:
            return -math.inf
        if isinstance(val, sympy.Integer):
            return int(val)
        # TODO: Remove this adjustment when fractional ranges are removed
        return adjust_func(val)

    range_constraints = get_range_constraints(g)

    self._cc._symbolic_guards = {
        str(k): RangeConstraint(
            _sympy_int_to_int(v.lower, math.ceil),
            _sympy_int_to_int(v.upper, math.floor),
        )
        for k, v in range_constraints.items()
    }

    ftype, loc = self._graph_to_function_meta(g)
    # TODO: The FuncOp constructor requires a context-manager context.
    # Fix upstream and then unnest.
    with loc:
        func = func_dialect.FuncOp(
            func_name,
            ftype,
            ip=self._m_ip,
            visibility=func_visibility,
        )
        func.attributes["torch.assume_strict_symbolic_shapes"] = UnitAttr.get()
        entry_block = Block.create_at_start(func.body, ftype.inputs)
    node_importer = GraphNodeImporter(
        self,
        self._c,
        self._cc,
        entry_block,
    )
    node_importer.import_nodes(
        g.nodes, import_symbolic_shape_expressions=import_symbolic_shape_expressions
    )
    self.symbol_table.insert(func)
    return func


def _patch_sympy_expr_to_semi_affine_expr(
    expr: sympy.Expr, symbols_map: Dict[str, AffineSymbolExpr]
) -> AffineExpr:
    """Translate sympy expressions to MLIR (semi-)affine expressions.

    Recursively traverse the sympy expr AST and build the affine expr.
    This is not a perfect translation. Sympy expressions are much more
    expressive and not as constrained as affine (linear) expressions are.
    However, for the most part, we don't need to support all of sympy.
    PyTorch only uses a subset of sympy for capturing and expressing
    symbolic shapes, and among what's supported, we expect the semi-affine
    expressions to be sufficient.
    """

    if isinstance(expr, sympy.Symbol):
        return symbols_map[str(expr)]
    elif isinstance(expr, (int, sympy.Integer)):
        return AffineConstantExpr.get(expr)
    # This handles both add (`s0 + c`) and subtract (`s0 - c`).
    # The expression is `sympy.Add` in both cases but with args
    # (s0, c) in first case and (s0, -c) in the second case.
    elif isinstance(expr, sympy.Add):
        affine_expr = AffineConstantExpr.get(0)
        for arg in expr.args:
            affine_expr = AffineAddExpr.get(
                affine_expr, sympy_expr_to_semi_affine_expr(arg, symbols_map)
            )
        return affine_expr
    elif isinstance(expr, sympy.Mul):
        affine_expr = AffineConstantExpr.get(1)
        for arg in expr.args:
            affine_expr = AffineMulExpr.get(
                affine_expr, sympy_expr_to_semi_affine_expr(arg, symbols_map)
            )
        return affine_expr
    elif isinstance(expr, sympy.Pow):
        base, exp = expr.args
        # Only integer exponent is supported
        # So, s1 ** s0 isn't allowed.
        assert isinstance(exp, (int, sympy.Integer))
        assert exp > 0, "Only positive exponents supported in sympy.Pow"
        affine_expr = AffineConstantExpr.get(1)
        for _ in range(exp):
            affine_expr = AffineMulExpr.get(
                affine_expr, sympy_expr_to_semi_affine_expr(base, symbols_map)
            )
        return affine_expr
    elif isinstance(expr, sympy.Mod):
        dividend, divisor = expr.args
        return AffineModExpr.get(
            sympy_expr_to_semi_affine_expr(dividend, symbols_map),
            sympy_expr_to_semi_affine_expr(divisor, symbols_map),
        )
    elif isinstance(expr, FloorDiv):
        dividend, divisor = expr.args
        return AffineFloorDivExpr.get(
            sympy_expr_to_semi_affine_expr(dividend, symbols_map),
            sympy_expr_to_semi_affine_expr(divisor, symbols_map),
        )
    else:
        raise NotImplementedError(
            f"Translation of sympy.Expr of type {type(expr)} not implemented yet."
        )


fx_importer.FxImporter.import_stateless_graph = _patch_import_stateless_graph
fx_importer.sympy_expr_to_semi_affine_expr = _patch_sympy_expr_to_semi_affine_expr

def stateless_fx_import(
    gm: torch.fx.GraphModule,
    output_type: Union[str, OutputType] = OutputType.RAW,
    fx_importer: Optional[FxImporter] = None,
    hooks: Optional[FxImporterHooks] = None,
    model_name: str = "main",
    enable_graph_printing: bool = False,
    enable_ir_printing: bool = False,
    import_symbolic_shape_expressions:bool = False,
):
    if enable_graph_printing:
        gm.print_readable()
    context = ir.Context()
    torch_d.register_dialect(context)
    if fx_importer is None:
        fx_importer = FxImporter(context=context, hooks=hooks)
    fx_importer.import_stateless_graph(gm.graph, func_name=model_name, import_symbolic_shape_expressions=import_symbolic_shape_expressions)
    return _module_lowering(
        enable_ir_printing, OutputType.get(output_type), fx_importer.module
    )
