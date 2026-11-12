"""Run status-save tests while leaving unrelated distributed processes alone."""

from pathlib import Path
import sys


class ScopedCleanup:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            if item.cls is not None and item.cls.__name__ == "CheckAndMakePathTest":
                # Each torchrun launch already manages its own children. The
                # original helper uses global pkill patterns across all jobs.
                item.cls._cleanup_stale_processes = staticmethod(lambda: None)


def main():
    # Import the installed wheel before pytest adds checkout paths to sys.path.
    import torch_npu
    import pytest

    root = Path(__file__).resolve().parents[3]
    if Path(torch_npu.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Run from outside the checkout using the installed wheel")
    test_file = root / "test/distributed/test_status_save.py"
    return pytest.main([str(test_file), *sys.argv[1:]], plugins=[ScopedCleanup()])


if __name__ == "__main__":
    sys.exit(main())
