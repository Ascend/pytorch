# Copyright (c) 2026, Huawei Technologies Co., Ltd

"""Registry for the torch_npu compatibility layer.

A compat point is one key plus one or more sides, each declaring when it
applies. Decorators only register; the module resolves at the end of its body:

    @compat_impl(key="patch_xxx", ge=(2, 16))
    def patch_xxx_upstream():
        return

    @compat_impl(key="patch_xxx", lt=(2, 16))
    def patch_xxx_local():
        ...

    patch_xxx = compat_impl_container["patch_xxx"].resolve()

The key groups the sides for every check and is expected to equal the name the
point is bound to.
"""

from typing import Any, Dict, List, Optional, Tuple

from torch_npu._compat.version import CURRENT_VERSION, MIN_SUPPORTED_VERSION

__all__ = ["compat_impl", "compat_impl_container", "CompatError"]

GE = "ge"
LT = "lt"
JUDGEMENT = "judgement"

Version = Tuple[int, int]


class CompatError(RuntimeError):
    """Raised when a compat point cannot be registered or resolved."""


def _is_versioned(side: "_Side") -> bool:
    return side.op in (GE, LT)


class _Side:
    """One side of a compat point."""

    __slots__ = ("key", "module", "fn", "op", "version", "matched")

    def __init__(self, key, module, fn, op, version, matched):
        self.key = key
        self.module = module
        self.fn = fn
        self.op = op
        self.version = version
        self.matched = matched

    def matches(self) -> bool:
        if self.op == JUDGEMENT:
            return self.matched
        if self.op == GE:
            return CURRENT_VERSION >= self.version
        return CURRENT_VERSION < self.version

    def describe(self) -> str:
        if self.op == JUDGEMENT:
            condition = f"judgement={self.matched}"
        else:
            symbol = ">=" if self.op == GE else "<"
            condition = f"{symbol} {self.version[0]}.{self.version[1]}"
        return f"{self.key} ({condition} -> {self.fn.__name__})"


class CompatPoint:
    """One key, with its sides kept in registration order."""

    def __init__(self, key: str, module: str):
        self.key = key
        self.module = module
        self._sides: List[_Side] = []

    def _add(self, side: _Side) -> None:
        if side.module != self.module:
            raise CompatError(
                f"compat point {self.key} is already registered by "
                f"{self.module}, cannot re-register from {side.module}"
            )
        if self._sides and _is_versioned(self._sides[0]) != _is_versioned(side):
            raise CompatError(
                f"compat point {self.key}: ge/lt and judgement cannot be mixed"
            )
        if _is_versioned(side):
            self._check_versioned(side)
        elif side.matched and any(existing.matched for existing in self._sides):
            raise CompatError(
                f"compat point {self.key}: more than one side matches"
            )
        self._sides.append(side)

    def _check_versioned(self, side: _Side) -> None:
        if side.version <= MIN_SUPPORTED_VERSION:
            raise CompatError(
                f"compat point {side.key}: threshold {side.version} is <= "
                f"MIN_SUPPORTED {MIN_SUPPORTED_VERSION}; this side should be removed"
            )
        for existing in self._sides:
            if existing.op == side.op:
                raise CompatError(
                    f"compat point {side.key}: duplicate {side.op} side"
                )
            if existing.version != side.version:
                raise CompatError(
                    f"compat point {side.key}: sides disagree on the threshold "
                    f"({existing.version} vs {side.version}), which would leave a "
                    f"gap or an overlap at the boundary"
                )
        if len(self._sides) >= 2:
            raise CompatError(f"compat point {side.key}: at most two versioned sides")

    def _pick(self) -> Optional[_Side]:
        for side in self._sides:
            if side.matches():
                return side
        return None

    def resolve(self) -> Any:
        """Return the matching side's function object without calling it."""
        side = self._pick()
        if side is None:
            raise CompatError(
                f"compat point {self.key} has no matching side for this version"
            )
        return side.fn

    def apply(self) -> Any:
        """Call the matching side with no arguments and return its value."""
        side = self._pick()
        if side is None:
            return None
        return side.fn()

    def describe_all(self) -> List[str]:
        return [side.describe() for side in self._sides]


class _Container:
    """Global registry, indexed by key."""

    def __init__(self) -> None:
        self._points: Dict[str, CompatPoint] = {}

    def __getitem__(self, key: str) -> CompatPoint:
        try:
            return self._points[key]
        except KeyError:
            raise CompatError(f"unknown compat point: {key}") from None

    def point(self, key: str, module: str) -> CompatPoint:
        existing = self._points.get(key)
        if existing is None:
            existing = CompatPoint(key, module)
            self._points[key] = existing
        return existing

    def keys(self) -> List[str]:
        return list(self._points)

    def _reset_for_test(self) -> None:
        """Test-only helper. Do not call it in normal runtime."""
        self._points.clear()


compat_impl_container = _Container()


def _validate(name, ge, lt, judgement):
    if not isinstance(name, str) or not name:
        raise CompatError("compat point key must be a non-empty string")
    if [ge is not None, lt is not None, judgement is not None].count(True) != 1:
        raise CompatError(
            f"compat point {name}: exactly one of ge / lt / judgement is required"
        )
    for kind, value in (("ge", ge), ("lt", lt)):
        if value is None:
            continue
        if (not isinstance(value, tuple) or len(value) != 2
                or not all(isinstance(part, int) for part in value)):
            raise CompatError(
                f"compat point {name}: {kind} must be an (int, int) version tuple"
            )


def compat_impl(*, key, ge=None, lt=None, judgement=None):
    """Register one side of a compat point.

    @compat_impl(key="x", ge=(2, 15))       # CURRENT_VERSION >= (2, 15)
    @compat_impl(key="x", lt=(2, 15))       # CURRENT_VERSION <  (2, 15)
    @compat_impl(key="x", judgement=<bool>)  # escape hatch for version-less cases
    """
    _validate(key, ge, lt, judgement)

    if judgement is not None:
        op, version, matched = JUDGEMENT, None, bool(judgement)
    elif ge is not None:
        op, version, matched = GE, ge, None
    elif lt is not None:
        op, version, matched = LT, lt, None
    else:
        # _validate() already guarantees exactly one of ge / lt / judgement.
        # Spell the missing case out so the invariant fails here with a clear
        # message instead of surfacing later as a TypeError from comparing
        # None against a version tuple.
        raise CompatError(f"compat point {key}: no side condition given")

    def decorator(fn):
        compat_impl_container.point(key, fn.__module__)._add(
            _Side(key, fn.__module__, fn, op, version, matched)
        )
        return fn

    return decorator
