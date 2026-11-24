"""Compare the precise test selection with the original strategy.

Both sides represent the full set of test files the PR pipeline would run:
the regular UT selection plus the independent inductor and distributed jobs.
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "ci"))

from access_control import TestMgr  # noqa: E402
from access_control.constants import (  # noqa: E402
    BASE_DIR,
    NOT_RUN_DIRECTLY,
    SLOW_TEST_BLOCKLIST,
    TEST_DIR,
)
from access_control.strategy import CoreTestStrategy, DirectoryMappingStrategy  # noqa: E402


def normalize(path):
    """Normalize a test path to an absolute path inside the repository."""
    p = path.strip()
    if "::" in p:
        p = p.split("::", 1)[0]
    if not p:
        return None
    p = Path(p)
    if p.is_absolute():
        return str(p)
    if p.parts and p.parts[0] == "test":
        return str(BASE_DIR / p)
    return str(TEST_DIR / p)


def load_precise(precise_dir):
    """Load the precise selection result; return None when status is not success."""
    status_file = precise_dir / "status"
    list_file = precise_dir / "test_files.txt"
    if not status_file.is_file():
        return None
    if status_file.read_text(encoding="utf-8").strip() != "success":
        return None
    files = set()
    if list_file.is_file():
        for line in list_file.read_text(encoding="utf-8").splitlines():
            normalized = normalize(line)
            if normalized:
                files.add(normalized)
    return files


def is_core(path):
    return path.startswith(str(BASE_DIR / "test/npu") + "/")


def apply_test_filters(files):
    """Apply the same existence and slow-test filters as TestMgr.analyze()."""
    result = set()
    for f in files:
        if not Path(f).exists():
            continue
        if any(name + ".py" in f for name in SLOW_TEST_BLOCKLIST):
            continue
        result.add(f)
    return result


def apply_job_filters(files, deleted_files):
    """Apply the UT job's post-selection exclusions to an arbitrary file set.

    access_control_test.py applies the not-run-directly, common_files.txt and
    version filters to every selected file, and ut_job removes some test files
    from the tree before selection. Precise files go through the same steps so
    both sides of the comparison describe the set the pipeline can actually run.
    """
    manager = TestMgr()
    manager.test_files["ut_files"] = sorted(files)
    manager.exclude_test_files(not_run_files=NOT_RUN_DIRECTLY, mode="not_run_directly")
    common_files = BASE_DIR / "common_files.txt"
    if common_files.is_file():
        manager.exclude_files_from_list(str(common_files))
    manager.filter_by_version(None, None)
    return {
        f for f in manager.get_test_files()["ut_files"]
        if normalize(f) not in deleted_files
    }


def read_modify_files(modify_files):
    """Read non-empty modify file paths; return [] when the file is missing."""
    path = Path(modify_files)
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_python_change(path):
    """DirectoryMapping only maps torch_npu paths; precise replaces it for Python files."""
    parts = Path(path).parts
    return bool(parts) and parts[0] == "torch_npu" and path.endswith(".py")


def check_inductor(files, target_branch, comment_body):
    """Mirror the check_inductor workflow decision."""
    if not files:
        return False
    if re.search(r"v2\.(12|13|14)\.0", target_branch):
        return False
    if "compile_inductor" in comment_body:
        return True
    return any(re.search(r"_inductor|/interface", f) for f in files)


def check_distribution(files):
    """Mirror the check_distribution workflow decision."""
    return any(re.search(r"/(distributed|rpc)", f) for f in files)


def inductor_files():
    """Inductor job file set: _inductor glob minus the filters the job applies.

    Mirrors access_control_test.py: after load_inductor_ut() it applies the
    not_run_directly and common_files.txt exclusions, then the version filter.
    selection_compare only runs against master, where the inductor job runs
    without --between_version, so files with @runIfVersion decorators are dropped.
    """
    manager = TestMgr()
    manager.test_files["ut_files"] = [
        str(path) for path in (TEST_DIR / "_inductor").rglob("test_*.py")
    ]
    manager.exclude_test_files(not_run_files=NOT_RUN_DIRECTLY, mode="not_run_directly")
    common_files = BASE_DIR / "common_files.txt"
    if common_files.is_file():
        manager.exclude_files_from_list(str(common_files))
    manager.filter_by_version(None, None)
    return set(manager.get_test_files()["ut_files"])


def real_ut_deleted_files():
    """Test files the regular UT job deletes from the tree before selection (ut_job.yml).

    The job runs `rm -rf test_jit.py test_jit*.py jit distributed/test_register_sharding.py
    distributed/fsdp2` inside test/ before access_control_test.py, so those files can
    never be selected by the real pipeline.
    """
    deleted = {str(p) for p in TEST_DIR.glob("test_jit*.py")}
    for rel in ("jit", "distributed/fsdp2"):
        deleted |= {str(p) for p in (TEST_DIR / rel).rglob("*.py")}
    deleted.add(str(TEST_DIR / "distributed" / "test_register_sharding.py"))
    return deleted


def distributed_deleted_files():
    """Test files the distributed job deletes before selection (ut_job_distributed.yml)."""
    names = (
        "test_allgather_base.py",
        "test_allgather_into_tensor.py",
        "test_allgather_togather.py",
        "test_reduce_scatter_base.py",
        "test_reduce_scatter_tensor.py",
        "test_register_sharding.py",
    )
    return {str(TEST_DIR / "distributed" / name) for name in names}


def distributed_files(modify_files):
    """Reproduce the distributed job's modify-file filtering and selection."""
    files = [f for f in read_modify_files(modify_files) if "distributed" in f or "rpc" in f]
    files.append("test/distributed/test_fault_mode.py")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(files) + "\n")
        tmp_path = fh.name
    try:
        return run_strategy(
            tmp_path, keep_core=False, deleted_files=distributed_deleted_files()
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def pipeline_extra_files(modify_files, target_branch, comment_body):
    """Test files the PR pipeline runs independently of the regular selection."""
    files = read_modify_files(modify_files)
    extra = set()
    if check_inductor(files, target_branch, comment_body):
        extra |= inductor_files()
    if check_distribution(files):
        extra |= distributed_files(modify_files)
    return extra


def run_strategy(modify_files, replace_directory_mapping=False, keep_core=True,
                 deleted_files=None):
    """Run the original selection with the same filters as the UT jobs.

    world_size is non-zero, matching the UT/get_count behavior that filters out
    test/_inductor changes handled by the dedicated inductor job. When
    modify_files.txt is absent, falls back to all core tests like
    access_control_test.py. deleted_files drops tests the target job removes
    from the tree before selection. The version filter mirrors the job, which on
    master runs without --between_version and drops @runIfVersion files.
    """
    manager = TestMgr()
    original_directory_mapping = DirectoryMappingStrategy.identify
    original_core = CoreTestStrategy.identify
    try:
        if not Path(modify_files).is_file():
            manager.load_core_ut()
        else:
            manager.load(modify_files, world_size=2)
            if replace_directory_mapping:
                def identify(self, modify_file):
                    if is_python_change(modify_file):
                        return []
                    return original_directory_mapping(self, modify_file)

                DirectoryMappingStrategy.identify = identify
            if not keep_core:
                CoreTestStrategy.identify = lambda self, modify_file: []

            manager.analyze()

        manager.exclude_test_files(not_run_files=NOT_RUN_DIRECTLY, mode="not_run_directly")
        common_files = BASE_DIR / "common_files.txt"
        if common_files.is_file():
            manager.exclude_files_from_list(str(common_files))
        manager.filter_by_version(None, None)
    finally:
        DirectoryMappingStrategy.identify = original_directory_mapping
        CoreTestStrategy.identify = original_core

    files = manager.get_test_files()["ut_files"]
    if deleted_files is None:
        return set(files)
    return {f for f in files if normalize(f) not in deleted_files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modify-files", default="modify_files.txt")
    parser.add_argument("--precise-dir", default="precise")
    parser.add_argument("--output", default="selection_comparison.txt")
    parser.add_argument("--target-branch", default="")
    parser.add_argument("--comment-body", default="")
    args = parser.parse_args()

    ut_deleted = real_ut_deleted_files()
    strategy_files = run_strategy(args.modify_files, deleted_files=ut_deleted)
    extra_files = pipeline_extra_files(args.modify_files, args.target_branch, args.comment_body)
    original_full = strategy_files | extra_files

    precise_raw = load_precise(Path(args.precise_dir))
    precise_files = (
        apply_job_filters(apply_test_filters(precise_raw), ut_deleted)
        if precise_raw is not None else None
    )

    lines = []
    if precise_files is None:
        precise_full = original_full
        formal_added = set()
        formal_removed = set()
        lines.append("precise status: failed or missing")
    else:
        precise_full = (
            precise_files
            | run_strategy(
                args.modify_files,
                replace_directory_mapping=True,
                deleted_files=ut_deleted,
            )
            | extra_files
        )
        formal_added = precise_full - original_full
        formal_removed = original_full - precise_full
        lines.append("precise status: success")
        lines.append(f"precise total (raw): {len(precise_raw)}")
        lines.append(f"precise total (filtered): {len(precise_files)}")

    both = precise_full & original_full
    precise_only = precise_full - original_full
    original_only = original_full - precise_full
    union = precise_full | original_full

    lines.extend([
        f"strategy total (regular): {len(strategy_files)}",
        f"pipeline extra files: {len(extra_files)}",
        f"original full total: {len(original_full)}",
        f"precise full total: {len(precise_full)}",
        f"both: {len(both)}",
        f"precise only: {len(precise_only)}",
        f"original only: {len(original_only)}",
        f"union total: {len(union)}",
        f"precise core files: {len([f for f in precise_full if is_core(f)])}",
        f"original core files: {len([f for f in original_full if is_core(f)])}",
        f"formal added: {len(formal_added)}",
        f"formal removed: {len(formal_removed)}",
        "",
        "=== precise (filtered) ===",
    ])
    lines.extend(sorted(precise_files or ()))
    lines.extend(["", "=== original full ==="])
    lines.extend(sorted(original_full))
    lines.extend(["", "=== precise full ==="])
    lines.extend(sorted(precise_full))
    lines.extend(["", "=== formal added ==="])
    lines.extend(sorted(formal_added))
    lines.extend(["", "=== formal removed ==="])
    lines.extend(sorted(formal_removed))

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
