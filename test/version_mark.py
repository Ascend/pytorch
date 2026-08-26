"""Version-range decorator for test cases.

The decorator serves two purposes:
1. Declaration tag for AST-based CI file-level filtering (CI scans for
   @runIfVersion decorators and drops files whose entire version range
   falls outside the CI target).
2. Runtime skip: if a decorated test case/class does NOT match the CI
   target range (set via :func:`set_ci_version_range` or environment
   variables ``CI_VERSION_MIN`` / ``CI_VERSION_MAX``), it is skipped via
   :func:`unittest.skip`. This allows per-test-case filtering for files
   that contain mixed version ranges.

Usage
-----
    from version_mark import runIfVersion

    @runIfVersion(min="2.10")
    def test_new_api(self): ...           # runs on >= 2.10

    @runIfVersion(min="2.10", max="2.12")
    def test_legacy(self): ...            # runs on 2.10~2.12

    @runIfVersion(max="2.9")
    def test_old_path(self): ...          # runs on <= 2.9

    @runIfVersion(min="2.11")
    class TestFSDP2(TestCase): ...        # class-level / all methods
"""

import os
import unittest
from typing import Callable, Optional, Tuple, Union


__all__ = ["runIfVersion", "set_ci_version_range"]


# CI target range; when both None the decorator does not skip anything.
_ci_min: Optional[Tuple[int, ...]] = None
_ci_max: Optional[Tuple[int, ...]] = None


def set_ci_version_range(
    min_ver: Optional[Tuple[int, ...]] = None,
    max_ver: Optional[Tuple[int, ...]] = None,
) -> None:
    """Set the CI target version range used by :func:`runIfVersion` at runtime.

    This is called from the CI entry point (``ci/access_control_test.py``)
    after parsing ``--between_version``.

    Args:
        min_ver: Lower bound tuple, e.g. ``(2, 10)``. ``None`` = no lower bound.
        max_ver: Upper bound tuple, e.g. ``(2, 12)``. ``None`` = no upper bound.
    """
    global _ci_min, _ci_max
    _ci_min = min_ver
    _ci_max = max_ver


def _parse_version_tuple(ver: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in ver.split("."))


def _env_version_tuple(name: str) -> Optional[Tuple[int, ...]]:
    """Read a version tuple from an environment variable."""
    val = os.getenv(name)
    if val:
        try:
            return _parse_version_tuple(val)
        except ValueError:
            pass
    return None


def _ranges_intersect(
    a_min: Optional[Tuple[int, ...]],
    a_max: Optional[Tuple[int, ...]],
    b_min: Optional[Tuple[int, ...]],
    b_max: Optional[Tuple[int, ...]],
) -> bool:
    """Check if range A intersects range B (both inclusive).

    ``None`` means unbounded on that side.
    """
    if (a_min is None and a_max is None) or (b_min is None and b_max is None):
        return True
    if a_max is not None and b_min is not None and a_max < b_min:
        return False
    if a_min is not None and b_max is not None and a_min > b_max:
        return False
    return True


def runIfVersion(
    min: Optional[str] = None,
    max: Optional[str] = None,
) -> Callable[[Union[Callable, type]], Union[Callable, type]]:
    """Decorator: restrict a test case/class to a specific version range.

    At decoration time the decorator compares the declared range against the
    CI target range (set via :func:`set_ci_version_range` or env vars). When
    they do not intersect the target is wrapped with :func:`unittest.skip`.

    Args:
        min: Minimum version (e.g. ``"2.10"``), inclusive.  ``None`` = no lower bound.
        max: Maximum version (e.g. ``"2.12"``), inclusive.  ``None`` = no upper bound.
    """
    if min is None and max is None:
        raise ValueError("runIfVersion requires at least one of `min` or `max`")

    dec_min = _parse_version_tuple(min) if min is not None else None
    dec_max = _parse_version_tuple(max) if max is not None else None

    # Resolve CI range: explicit module-level setting wins over env vars.
    ci_min = _ci_min if _ci_min is not None else _env_version_tuple("CI_VERSION_MIN")
    ci_max = _ci_max if _ci_max is not None else _env_version_tuple("CI_VERSION_MAX")

    should_skip = False
    if ci_min is not None or ci_max is not None:
        if not _ranges_intersect(dec_min, dec_max, ci_min, ci_max):
            should_skip = True

    def decorator(target):
        if should_skip:
            ci_min_str = ".".join(map(str, ci_min)) if ci_min else "-inf"
            ci_max_str = ".".join(map(str, ci_max)) if ci_max else "+inf"
            reason = (
                f"runIfVersion: test range [{min or '0'}, {max or 'inf'}] "
                f"does not intersect CI range [{ci_min_str}, {ci_max_str}]"
            )
            return unittest.skip(reason)(target)
        return target

    return decorator