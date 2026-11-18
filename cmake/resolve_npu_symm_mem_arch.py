"""Resolve one architecture using an explicitly supplied torch-npu helper.

Usage: resolve_npu_symm_mem_arch.py TORCH_NPU_SOURCE_DIR_OR_HELPER ARCH
No torch_npu package import or device probing is performed. An empty ARCH
uses resolve_npu_arch's explicit environment-variable policy.
"""

from pathlib import Path
import re
import runpy
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: resolve_npu_symm_mem_arch.py TORCH_NPU_SOURCE_DIR_OR_HELPER ARCH")
    extension = Path(sys.argv[1])
    if extension.is_dir():
        extension = extension / "torch_npu/utils/_npu_extension.py"
    try:
        resolve_npu_arch = runpy.run_path(str(extension))["resolve_npu_arch"]
        architecture = resolve_npu_arch(explicit=sys.argv[2] or None)
        # Validate command/path syntax only; ASC checks primitive capabilities.
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", architecture) is None:
            raise ValueError(f"Invalid NPU architecture token: {architecture!r}")
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        sys.exit(f"NPU_SYMM_MEM_ARCH: {error}")
    print(architecture)


if __name__ == "__main__":
    main()
