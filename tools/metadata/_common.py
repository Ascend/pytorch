"""Shared helpers for the tools/metadata dynamic-metadata providers.

Sibling modules import this at top level: scikit-build-core resolves a
provider's imports through a meta-path finder that is only active while the
provider module itself is being imported, so a deferred (in-function) import
of this module would not resolve.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_version_fn():
    spec = importlib.util.spec_from_file_location(
        "torch_npu_version",
        _ROOT / "tools" / "setup_helpers" / "version.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load setup_helpers/version.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_version


def get_version() -> str:
    """Resolve the base version (TORCH_VERSION or the installed PyTorch version)."""
    return _load_version_fn()()


def get_package_version() -> str:
    """Resolve the full package version, mirroring setup.py's
    generate_torch_npu_version(): the base version plus a ``+git<sha7>`` local
    tag unless BUILD_WITHOUT_SHA is set."""
    version = get_version()
    if os.environ.get("BUILD_WITHOUT_SHA") is not None:
        return version
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT)
            .decode("ascii")
            .strip()
        )
    except Exception:
        sha = "Unknown"
    return version + "+git" + sha[:7]
