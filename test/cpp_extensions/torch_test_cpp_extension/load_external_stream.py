import importlib
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_PACKAGES_DIR = REPO_ROOT / "build" / "packages"
CPP_EXTENSIONS_DIR = REPO_ROOT / "test" / "cpp_extensions"
MODULE_NAME = "torch_test_cpp_extension.npu_external_stream"


def _build_pythonpath_for_subprocess():
    parts = []
    if BUILD_PACKAGES_DIR.exists():
        parts.append(str(BUILD_PACKAGES_DIR))
    existing = os.environ.get("PYTHONPATH")
    if existing:
        filtered = [p for p in existing.split(os.pathsep) if p != str(REPO_ROOT)]
        parts.extend(filtered)
    return os.pathsep.join(parts)


def _build_extension_inplace():
    env = os.environ.copy()
    pythonpath = _build_pythonpath_for_subprocess()
    if pythonpath:
        env["PYTHONPATH"] = pythonpath

    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=str(CPP_EXTENSIONS_DIR),
        check=True,
        env=env,
    )


def load_external_stream_extension():
    for path in ("", str(REPO_ROOT)):
        while path in sys.path:
            sys.path.remove(path)

    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    import torch  # noqa: F401

    import torch_npu  # noqa: F401

    try:
        return importlib.import_module(MODULE_NAME)
    except ImportError:
        _build_extension_inplace()
        sys.modules.pop(MODULE_NAME, None)
        importlib.invalidate_caches()
        return importlib.import_module(MODULE_NAME)