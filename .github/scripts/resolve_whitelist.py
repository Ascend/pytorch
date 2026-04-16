#!/usr/bin/env python3
"""
Resolve whitelist/blacklist/CRASHED configurations to test include/exclude lists.

This script bridges case_paths_ci.yml (torch-npu's test selection) with
PyTorch's run_test.py (which uses --include/--exclude with TESTS list names).

Output:
  - include_tests.txt: test names for run_test.py --include (in TESTS format)
  - exclude_tests.txt: test names for run_test.py --exclude (in TESTS format)
  - extra_pytest_tests.txt: file paths for direct pytest (not in TESTS list)

TESTS format: no 'test/' prefix, no '.py' suffix. E.g.:
  test_autograd, distributed/test_c10d, nn/test_convolution
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# YAML loading (with fallback to manual parser if pyyaml not installed)
# ---------------------------------------------------------------------------

def _parse_simple_yaml_lists(raw_text: str) -> Dict:
    """Minimal YAML list parser for whitelist/blacklist format."""
    parsed = {"whitelist": [], "blacklist": [], "crashed_files": []}
    current_key = None
    for raw_line in raw_text.splitlines():
        without_comment = raw_line.split("#", 1)[0].rstrip()
        if not without_comment.strip():
            continue
        stripped = without_comment.lstrip()
        if not raw_line.startswith((" ", "\t")) and stripped.endswith(":"):
            key = stripped[:-1].strip()
            current_key = key if key in parsed else None
            continue
        if current_key and stripped.startswith("- "):
            value = stripped[2:].strip().strip("\"'")
            if value:
                parsed[current_key].append(value)
    return parsed


def load_yaml_config(yaml_path: str) -> Dict:
    """Load YAML configuration file."""
    if not os.path.exists(yaml_path):
        print(f"[resolve] Warning: YAML file not found: {yaml_path}")
        return {}
    raw = Path(yaml_path).read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(raw) or {}
    return _parse_simple_yaml_lists(raw)


# ---------------------------------------------------------------------------
# Test file scanning
# ---------------------------------------------------------------------------

def scan_test_py_files(test_dir: str) -> Set[str]:
    """Scan test_*.py files, return paths relative to test_dir (e.g. 'distributed/test_c10d.py')."""
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"[resolve] Error: Test directory not found: {test_dir}")
        sys.exit(1)
    files = set()
    for py_file in test_path.rglob("test_*.py"):
        files.add(str(py_file.relative_to(test_path)))
    return files


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def normalize_pattern(pattern: str) -> str:
    """Remove leading 'test/' prefix from a pattern."""
    p = pattern.strip()
    if p.startswith("test/"):
        p = p[5:]
    return p.rstrip("/")


def match_files(pattern: str, all_py_files: Set[str]) -> Set[str]:
    """Match a whitelist/blacklist/crashed pattern against .py files.

    Pattern types:
      - 'test/test_utils.py'      -> exact file match
      - 'test/distributed'         -> all test_*.py under that directory
      - 'test/nn/test_conv*.py'    -> glob (not currently used, kept for safety)
    """
    norm = normalize_pattern(pattern)
    if not norm:
        return set()

    if norm.endswith(".py"):
        # Exact file
        return {norm} if norm in all_py_files else set()
    else:
        # Directory prefix: match all .py files under it
        prefix = norm + "/"
        return {f for f in all_py_files if f.startswith(prefix)}


# ---------------------------------------------------------------------------
# Core resolution
# ---------------------------------------------------------------------------

def resolve(
    case_paths: Dict,
    crashed: Dict,
    all_py_files: Set[str],
) -> Tuple[Set[str], Set[str]]:
    """Apply whitelist -> blacklist -> crashed filtering.

    Returns (included_files, excluded_files) — both are sets of relative .py paths.
    """
    whitelist = case_paths.get("whitelist", [])
    blacklist = case_paths.get("blacklist", [])
    crashed_files = crashed.get("crashed_files", [])

    # Step 1: whitelist
    included = set()
    for pat in whitelist:
        included |= match_files(pat, all_py_files)

    # Step 2: subtract blacklist
    bl = set()
    for pat in blacklist:
        bl |= match_files(pat, all_py_files)

    # Step 3: subtract crashed
    cr = set()
    for pat in crashed_files:
        cr |= match_files(pat, all_py_files)

    excluded = bl | cr
    included -= excluded
    return included, excluded


# ---------------------------------------------------------------------------
# TESTS list integration
# ---------------------------------------------------------------------------

def to_tests_format(py_path: str) -> str:
    """Convert relative .py path to TESTS format (no .py, no test/ prefix).

    'distributed/test_c10d.py' -> 'distributed/test_c10d'
    'test_autograd.py'         -> 'test_autograd'
    """
    if py_path.endswith(".py"):
        py_path = py_path[:-3]
    return py_path


def load_tests_list(pytorch_root: str) -> Set[str]:
    """Import TESTS from discover_tests.py in the PyTorch source tree."""
    if not pytorch_root or not Path(pytorch_root).exists():
        return set()

    sys.path.insert(0, str(pytorch_root))
    try:
        from tools.testing.discover_tests import TESTS
        return set(TESTS)
    except Exception as e:
        print(f"[resolve] Warning: Could not load TESTS list: {e}")
        return set()
    finally:
        if str(pytorch_root) in sys.path:
            sys.path.remove(str(pytorch_root))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Resolve case_paths_ci.yml + CRASHED.yml -> run_test.py include/exclude lists"
    )
    parser.add_argument("--case-paths-config", required=True)
    parser.add_argument("--crashed-files-config", required=True)
    parser.add_argument("--test-dir", required=True, help="PyTorch test/ directory")
    parser.add_argument("--pytorch-root", default=None, help="PyTorch source root (for TESTS list)")
    parser.add_argument("--output-include", required=True)
    parser.add_argument("--output-exclude", required=True)
    parser.add_argument("--output-extra", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load configs
    case_paths = load_yaml_config(args.case_paths_config)
    crashed = load_yaml_config(args.crashed_files_config)
    if not case_paths:
        print("[resolve] Error: empty case_paths_config")
        sys.exit(1)

    # Scan only .py test files (no directories)
    all_py_files = scan_test_py_files(args.test_dir)
    print(f"[resolve] Scanned {len(all_py_files)} test_*.py files")

    # Resolve
    included, excluded = resolve(case_paths, crashed, all_py_files)
    print(f"[resolve] After whitelist-blacklist-crashed: include={len(included)}, exclude={len(excluded)}")

    # Load TESTS list
    tests_list = load_tests_list(args.pytorch_root) if args.pytorch_root else set()
    if tests_list:
        print(f"[resolve] Loaded {len(tests_list)} entries from TESTS list")

    # Split included files: those in TESTS go to --include, rest to extra pytest
    include_for_run_test: List[str] = []
    extra_for_pytest: List[str] = []

    for py_file in sorted(included):
        name = to_tests_format(py_file)
        if tests_list and name in tests_list:
            include_for_run_test.append(name)
        elif not tests_list:
            # TESTS list unavailable (import failed): assume all go to run_test.py
            # This is a fallback — normally pytorch_root is always provided
            include_for_run_test.append(name)
        else:
            # Not in TESTS list, run_test.py --include will reject it
            # -> run via direct pytest; needs test/ prefix for cwd=repo_root
            extra_for_pytest.append(f"test/{py_file}")

    # Exclude list: only include names that are in TESTS list
    # run_test.py --exclude uses choices=TESTS (strict), so passing names
    # not in TESTS will cause argparse to error out
    exclude_for_run_test: List[str] = []
    for f in sorted(excluded):
        name = to_tests_format(f)
        if not tests_list or name in tests_list:
            exclude_for_run_test.append(name)

    # Write outputs
    Path(args.output_include).write_text("\n".join(include_for_run_test) + "\n" if include_for_run_test else "")
    Path(args.output_exclude).write_text("\n".join(exclude_for_run_test) + "\n" if exclude_for_run_test else "")
    Path(args.output_extra).write_text("\n".join(extra_for_pytest) + "\n" if extra_for_pytest else "")

    print(f"[resolve] Output: include={len(include_for_run_test)}, exclude={len(exclude_for_run_test)}, extra={len(extra_for_pytest)}")

    if args.verbose:
        print("\n=== Include (run_test.py --include, first 20) ===")
        for n in include_for_run_test[:20]:
            print(f"  {n}")
        if len(include_for_run_test) > 20:
            print(f"  ... and {len(include_for_run_test) - 20} more")

        print("\n=== Extra (direct pytest, first 20) ===")
        for p in extra_for_pytest[:20]:
            print(f"  {p}")
        if len(extra_for_pytest) > 20:
            print(f"  ... and {len(extra_for_pytest) - 20} more")


if __name__ == "__main__":
    main()
