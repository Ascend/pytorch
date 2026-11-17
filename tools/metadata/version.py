"""Dynamic-metadata provider for the version field.

Delegates to tools/setup_helpers/version.py (via _common), reproducing the
setup.py generate_torch_npu_version() version: base version plus a ``+git<sha7>``
local tag unless BUILD_WITHOUT_SHA is set.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

# _common is resolved at build time via scikit-build-core's provider path,
# not statically importable from the repo root.
from _common import get_package_version

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

    return {"version": get_package_version()}
