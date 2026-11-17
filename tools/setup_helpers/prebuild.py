#!/usr/bin/env python3
"""Pre-configure steps previously run by setup.py before CMake configure.

Consolidates the codegen/stub work that setup.py used to perform outside of
CMake so that scikit-build-core can drive the whole build through CMake:

- generate_torch_npu_version(): writes torch_npu/version.py.
- generate_bindings_code(): fetches ACL headers and runs the torchnpugen/op-plugin
  codegen via generate_code.sh.
- build_stub(): builds the ACL stub .so libraries (libhccl.so, libascendcl.so, ...).

Invoked by cmake/Setup.cmake:

- ``--version`` prints the package version and exits (cheap; used to populate
  TORCH_VERSION).
- otherwise it runs the full pre-configure sequence. It must run before CMake's
  configure-time ``FILE(GLOB ...)`` collects the generated sources.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# tools/ is a namespace package; add the repo root so
# ``tools.setup_helpers.version`` resolves.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.setup_helpers.version import get_version  # noqa: E402

UNKNOWN = "Unknown"


def get_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return UNKNOWN


def get_package_version() -> str:
    """Mirror setup.py generate_torch_npu_version(): base version plus a
    ``+git<sha7>`` local tag unless BUILD_WITHOUT_SHA is set."""
    version = get_version()
    if os.environ.get("BUILD_WITHOUT_SHA") is None:
        version += "+git" + get_sha()[:7]
    return version


def generate_version_file() -> None:
    version_path = BASE_DIR / "torch_npu" / "version.py"
    if version_path.exists():
        version_path.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    sha = get_sha()
    version = get_package_version()
    with os.fdopen(os.open(version_path, flags, modes), "w") as f:
        f.write("__version__ = '{version}'\n".format(version=version))
        f.write("git_version = {}\n".format(repr(sha)))
    os.chmod(version_path, stat.S_IRUSR | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)


def generate_bindings_code(version: str) -> None:
    cmd = ["bash", str(BASE_DIR / "generate_code.sh"), sys.executable, version]
    if subprocess.call(cmd) != 0:
        print(f"Failed to generate ATEN bindings: {cmd}", file=sys.stderr)
        sys.exit(1)


def build_stub(base_dir: Path) -> None:
    cmd = ["sh", str(base_dir / "third_party" / "acl" / "libs" / "build_stub.sh")]
    if subprocess.call(cmd) != 0:
        print(f"Failed to build stub: {cmd}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    version = get_package_version()
    generate_version_file()
    generate_bindings_code(version)
    tensorpipe = BASE_DIR / "third_party" / "Tensorpipe"
    if (tensorpipe / "third_party" / "acl" / "libs").exists():
        build_stub(tensorpipe)
    build_stub(BASE_DIR)


if __name__ == "__main__":
    if "--version" in sys.argv:
        print(get_package_version())
    else:
        main()
