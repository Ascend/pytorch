"""Dynamic-metadata provider for the dependencies field.

Emits ``torch==<base>.*`` (pin the base PyTorch release line) plus ``numpy``,
matching the legacy setup.py ``_get_torch_requires()``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

# _common is resolved at build time via scikit-build-core's provider path,
# not statically importable from the repo root.
from _common import get_version
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["dynamic_metadata"]


def dynamic_metadata(
    settings: Mapping[str, Any],
    project: Mapping[str, Any],
) -> dict[str, Any]:
    if settings:
        msg = f"This provider takes no settings, got {sorted(settings)}"
        raise RuntimeError(msg)

    try:
        base_version = Version(get_version()).base_version
    except Exception:
        base_version = get_version().split("+")[0].split(".post")[0].split(".dev")[0]
    # The trailing ".*" turns the exact match into a PEP 440 prefix match,
    # letting pip honor an already-installed nightly/rc instead of failing
    # to find the not-yet-published base release on the index.
    return {"dependencies": [f"torch=={base_version}.*", "numpy"]}
