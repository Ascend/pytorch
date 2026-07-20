"""Version-range decorator for test cases.

Acts as a declaration tag for AST-based CI filtering. The decorator itself
does NOT skip anything at runtime; the CI pipeline (via --between_version)
scans test files for @runIfVersion decorators and only runs files whose
declared range intersects the CI range.

Usage
-----
    from test.utils.version_mark import runIfVersion

    @runIfVersion(min="2.10")
    def test_new_api(self): ...           # declares: runs on >= 2.10

    @runIfVersion(min="2.10", max="2.12")
    def test_legacy(self): ...            # declares: runs on 2.10~2.12

    @runIfVersion(max="2.9")
    def test_old_path(self): ...          # declares: runs on <= 2.9

    @runIfVersion(min="2.11")
    class TestFSDP2(TestCase): ...        # class-level declaration
"""

from typing import Callable, Optional, Union


__all__ = ["runIfVersion"]


def runIfVersion(
    min: Optional[str] = None,
    max: Optional[str] = None,
) -> Callable[[Union[Callable, type]], Union[Callable, type]]:
    """Decorator: declare a version range for a test (inclusive bounds).

    This is a declaration tag only; the decorator does not skip anything at
    runtime. Test selection is handled at CI time by file-level filtering
    (see ci/access_control/version_filter.py).

    Args:
        min: Minimum version (e.g. "2.10"), inclusive. None = no lower bound.
        max: Maximum version (e.g. "2.12"), inclusive. None = no upper bound.
    """
    if min is None and max is None:
        raise ValueError("runIfVersion requires at least one of `min` or `max`")

    def decorator(target):
        return target

    return decorator
