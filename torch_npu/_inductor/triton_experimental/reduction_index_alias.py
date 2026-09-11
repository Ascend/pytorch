# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Structured aliases for reduction indices rewritten by range-tree folding.

The range-tree folds below can replace a flat reduction index with a set of
independent range-tree nodes.  The original range-tree entry is still emitted
by upstream codegen, so the alias metadata has two jobs:

* retain the alias expression as SymPy until the backend printer owns it; and
* attach the alias explicitly to every emitted reduction-loop pass that uses it.

Aliases are never inferred from generated Triton source.  An original range-tree
assignment is represented by ``ReductionIndexAssignmentLine`` and suppressed by
its structured index identity after folding selects a replacement expression.
"""

from dataclasses import dataclass, replace
from graphlib import TopologicalSorter
from typing import Callable, Iterable, Mapping

import sympy
from torch._inductor.codegen.common import DeferredLineBase


class ReductionIndexAliasError(ValueError):
    pass


@dataclass(frozen=True)
class ReductionIndexAlias:
    name: str
    expr: sympy.Expr
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class ReductionLoopPass:
    """The IR visibility contract for exactly one emitted reduction pass."""

    scope_id: int
    reduction_prefixes: tuple[str, ...]
    reduction_index_aliases: tuple[ReductionIndexAlias, ...] = ()


class ReductionIndexAssignmentState:
    """Track original range-tree assignments superseded by folded aliases."""

    def __init__(self):
        self._suppressed: dict[str, set[str]] = {}

    def suppress_assignments(
        self, reduction_prefix: str, index_names: Iterable[str]
    ) -> None:
        self._suppressed.setdefault(reduction_prefix, set()).update(index_names)

    def is_assignment_suppressed(
        self, reduction_prefix: str, index_name: str
    ) -> bool:
        return index_name in self._suppressed.get(reduction_prefix, ())


class ReductionIndexAssignmentLine(DeferredLineBase):
    """A range-tree assignment with identity retained outside generated text."""

    def __init__(
        self,
        state: ReductionIndexAssignmentState,
        reduction_prefix: str,
        index_name: str,
        line: str,
    ):
        super().__init__(line)
        self._state = state
        self.reduction_prefix = reduction_prefix
        self.index_name = index_name

    def __call__(self) -> str | None:
        if self._state.is_assignment_suppressed(
            self.reduction_prefix, self.index_name
        ):
            return None
        return self.line

    def _new_line(self, line: str):
        return type(self)(
            self._state, self.reduction_prefix, self.index_name, line
        )


def make_reduction_loop_pass(
    scope_id: int, reduction_prefixes: Iterable[str]
) -> ReductionLoopPass:
    return ReductionLoopPass(
        scope_id=scope_id, reduction_prefixes=tuple(reduction_prefixes)
    )


def build_reduction_index_aliases(
    alias_exprs: Mapping[str, sympy.Expr],
) -> tuple[ReductionIndexAlias, ...]:
    alias_names = set(alias_exprs)
    aliases: dict[str, ReductionIndexAlias] = {}
    for name, expr in sorted(alias_exprs.items()):
        try:
            expr = sympy.sympify(expr)
        except (TypeError, ValueError, sympy.SympifyError) as exc:
            raise ReductionIndexAliasError(
                f"invalid reduction index alias {name}: {expr}"
            ) from exc
        dependencies = tuple(
            sorted(
                str(symbol)
                for symbol in expr.free_symbols
                if str(symbol) in alias_names and str(symbol) != name
            )
        )
        aliases[name] = ReductionIndexAlias(name, expr, dependencies)
    try:
        order = tuple(
            TopologicalSorter(
                {name: set(alias.dependencies) for name, alias in aliases.items()}
            ).static_order()
        )
    except ValueError as exc:
        raise ReductionIndexAliasError("cyclic reduction index aliases") from exc
    return tuple(aliases[name] for name in order)


def validate_reduction_index_alias_scope(
    aliases: Iterable[ReductionIndexAlias], visible_symbols: Iterable[object]
) -> None:
    visible = {str(name) for name in visible_symbols}
    for alias in aliases:
        unknown = {
            str(symbol)
            for symbol in alias.expr.free_symbols
            if str(symbol) not in visible
        }
        if unknown:
            raise ReductionIndexAliasError(
                f"{alias.name} uses undefined symbols: {sorted(unknown)}"
            )
        visible.add(alias.name)


def attach_reduction_index_aliases(
    loop_passes: Iterable[ReductionLoopPass],
    reduction_prefix: str,
    aliases: Iterable[ReductionIndexAlias],
) -> tuple[ReductionLoopPass, ...]:
    """Attach aliases to every loop pass containing ``reduction_prefix``.

    A pass may contain more than one reduction tree.  Preserve existing aliases
    and reject a conflicting duplicate instead of silently choosing one plan.
    """

    aliases = tuple(aliases)
    updated_passes = []
    for reduction_pass in loop_passes:
        if reduction_prefix not in reduction_pass.reduction_prefixes:
            updated_passes.append(reduction_pass)
            continue
        merged = {
            alias.name: alias
            for alias in reduction_pass.reduction_index_aliases
        }
        for alias in aliases:
            existing = merged.get(alias.name)
            if existing is not None and existing != alias:
                raise ReductionIndexAliasError(
                    f"conflicting reduction index alias {alias.name} "
                    f"in reduction scope {reduction_pass.scope_id}"
                )
            merged[alias.name] = alias
        order = [
            alias.name for alias in reduction_pass.reduction_index_aliases
        ]
        order.extend(alias.name for alias in aliases if alias.name not in order)
        updated_passes.append(
            replace(
                reduction_pass,
                reduction_index_aliases=tuple(merged[name] for name in order),
            )
        )
    return tuple(updated_passes)


def superseded_reduction_index_names(
    reduction_pass: ReductionLoopPass,
    generated_indices: Iterable[str],
) -> frozenset[str]:
    """Indices whose original range-tree assignments are superseded.

    The loop rewrite generates the promoted leaf indices itself and emits each
    reduction index alias from ``reduction_index_aliases``. Keeping this set in
    terms of structured identities avoids recognizing either kind of assignment
    from its rendered Triton source.
    """

    return frozenset(generated_indices).union(
        alias.name for alias in reduction_pass.reduction_index_aliases
    )


def render_reduction_index_aliases(
    reduction_pass: ReductionLoopPass,
    render_expr: Callable[[sympy.Expr], str],
    indent: str,
) -> list[str]:
    return [
        f"{indent}{alias.name} = {render_expr(alias.expr)}"
        for alias in reduction_pass.reduction_index_aliases
    ]
