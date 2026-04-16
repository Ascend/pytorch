#!/usr/bin/env python3
"""
Resolve whitelist/blacklist/CRASHED configurations to test include/exclude lists.

This script:
1. Loads whitelist/blacklist from case_paths_ci.yml
2. Loads crashed files from CRASHED.yml
3. Matches whitelist patterns to actual test files
4. Filters out blacklist and crashed files
5. Separates files into:
   - include_tests.txt: files in TESTS list (for run_test.py --include)
   - exclude_tests.txt: files to exclude (for run_test.py --exclude)
   - extra_pytest_tests.txt: files NOT in TESTS list (for direct pytest execution)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml


def load_yaml_config(yaml_path: str) -> Dict:
    """Load YAML configuration file."""
    if not os.path.exists(yaml_path):
        print(f"[resolve_whitelist] Warning: YAML file not found: {yaml_path}")
        return {}

    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_all_test_files(test_dir: str) -> Set[str]:
    """Get all test_*.py files from test directory, relative to test_dir."""
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"[resolve_whitelist] Error: Test directory not found: {test_dir}")
        sys.exit(1)

    test_files = set()
    for py_file in test_path.rglob("test_*.py"):
        # Get relative path from test_dir, e.g., "test/test_utils.py" -> "test_utils.py"
        rel_path = str(py_file.relative_to(test_path))
        test_files.add(rel_path)

    # Also include directories that contain tests (for pattern matching)
    for item in test_path.rglob("*"):
        if item.is_dir():
            rel_path = str(item.relative_to(test_path))
            # Skip hidden directories and __pycache__
            if not rel_path.startswith(".") and "__pycache__" not in rel_path:
                test_files.add(rel_path)

    return test_files


def match_pattern(pattern: str, test_files: Set[str]) -> Set[str]:
    """Match a whitelist/blacklist pattern against test files.

    Patterns can be:
    - test/test_utils.py -> exact match
    - test/distributed -> matches all files under this directory
    """
    matched = set()

    # Normalize pattern (remove leading test/ if present)
    normalized_pattern = pattern
    if pattern.startswith("test/"):
        normalized_pattern = pattern[5:]

    # Check if it's a directory pattern (no .py extension)
    if not normalized_pattern.endswith(".py"):
        # Directory pattern: match all files under this directory
        for f in test_files:
            if f.startswith(normalized_pattern + "/") or f == normalized_pattern:
                matched.add(f)
    else:
        # File pattern: exact match
        if normalized_pattern in test_files:
            matched.add(normalized_pattern)

    return matched


def resolve_test_lists(
    case_paths_config: Dict,
    crashed_config: Dict,
    test_files: Set[str],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Resolve whitelist/blacklist/crashed to final test lists.

    Returns:
        - include_set: files to include (whitelist - blacklist - crashed)
        - exclude_set: files to exclude (blacklist + crashed)
        - crashed_set: files that crashed
    """
    whitelist = case_paths_config.get("whitelist", [])
    blacklist = case_paths_config.get("blacklist", [])
    crashed_files = crashed_config.get("crashed_files", [])

    # Match whitelist patterns
    whitelist_matched = set()
    for pattern in whitelist:
        matched = match_pattern(pattern, test_files)
        whitelist_matched.update(matched)

    # Match blacklist patterns
    blacklist_matched = set()
    for pattern in blacklist:
        matched = match_pattern(pattern, test_files)
        blacklist_matched.update(matched)

    # Match crashed files
    crashed_matched = set()
    for pattern in crashed_files:
        matched = match_pattern(pattern, test_files)
        crashed_matched.update(matched)

    # Final include set = whitelist - blacklist - crashed
    include_set = whitelist_matched - blacklist_matched - crashed_matched

    # Exclude set = blacklist + crashed (for run_test.py --exclude)
    exclude_set = blacklist_matched | crashed_matched

    return include_set, exclude_set, crashed_matched


def load_tests_list(pytorch_root: str) -> Set[str]:
    """Load TESTS list from PyTorch discover_tests.py.

    This requires importing from the PyTorch source tree.
    """
    tools_path = Path(pytorch_root) / "tools" / "testing"
    if not tools_path.exists():
        print(f"[resolve_whitelist] Warning: tools/testing path not found: {pytorch_root}")
        return set()

    # Temporarily add pytorch_root to sys.path for import
    sys.path.insert(0, str(pytorch_root))
    try:
        from tools.testing.discover_tests import TESTS
        tests_set = set(TESTS)
        print(f"[resolve_whitelist] Loaded {len(tests_set)} tests from TESTS list")
        return tests_set
    except ImportError as e:
        print(f"[resolve_whitelist] Warning: Could not import TESTS: {e}")
        return set()
    finally:
        sys.path.pop(0)


def format_for_run_test_py(test_name: str) -> str:
    """Format test name for run_test.py.

    run_test.py expects test names without test/ prefix and without .py suffix.
    E.g., "test/test_utils.py" -> "test_utils"
    """
    # Remove test/ prefix if present
    if test_name.startswith("test/"):
        test_name = test_name[5:]

    # Remove .py suffix if present
    if test_name.endswith(".py"):
        test_name = test_name[:-3]

    return test_name


def main():
    parser = argparse.ArgumentParser(description="Resolve test whitelist/blacklist configurations")
    parser.add_argument(
        "--case-paths-config",
        required=True,
        help="Path to case_paths_ci.yml",
    )
    parser.add_argument(
        "--crashed-files-config",
        required=True,
        help="Path to CRASHED.yml",
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        help="Path to PyTorch test directory",
    )
    parser.add_argument(
        "--pytorch-root",
        default=None,
        help="Path to PyTorch source root (for loading TESTS list)",
    )
    parser.add_argument(
        "--output-include",
        required=True,
        help="Output file for include tests",
    )
    parser.add_argument(
        "--output-exclude",
        required=True,
        help="Output file for exclude tests",
    )
    parser.add_argument(
        "--output-extra",
        required=True,
        help="Output file for extra pytest tests",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information",
    )

    args = parser.parse_args()

    # Load configurations
    case_paths_config = load_yaml_config(args.case_paths_config)
    crashed_config = load_yaml_config(args.crashed_files_config)

    if not case_paths_config:
        print("[resolve_whitelist] Error: No whitelist/blacklist found")
        sys.exit(1)

    # Get all test files
    test_files = get_all_test_files(args.test_dir)
    print(f"[resolve_whitelist] Found {len(test_files)} test files/directories")

    # Resolve test lists
    include_set, exclude_set, crashed_set = resolve_test_lists(
        case_paths_config,
        crashed_config,
        test_files,
    )

    print(f"[resolve_whitelist] Whitelist matched: {len(include_set) + len(exclude_set) - len(crashed_set)} files")
    print(f"[resolve_whitelist] Blacklist matched: {len(exclude_set - crashed_set)} files")
    print(f"[resolve_whitelist] Crashed files: {len(crashed_set)} files")
    print(f"[resolve_whitelist] Final include: {len(include_set)} files")

    # Load TESTS list if pytorch_root provided
    tests_list = set()
    if args.pytorch_root:
        tests_list = load_tests_list(args.pytorch_root)

    # Separate into run_test.py format and extra pytest format
    include_for_run_test = []
    extra_for_pytest = []

    for test_file in sorted(include_set):
        formatted_name = format_for_run_test_py(test_file)

        if tests_list and formatted_name in tests_list:
            include_for_run_test.append(formatted_name)
        else:
            # Files not in TESTS list need to be run with direct pytest
            # These need full path: test/test_file.py
            extra_for_pytest.append(f"test/{test_file}")

    # Format exclude list for run_test.py
    exclude_for_run_test = []
    for test_file in sorted(exclude_set):
        formatted_name = format_for_run_test_py(test_file)
        exclude_for_run_test.append(formatted_name)

    # Write output files
    with open(args.output_include, "w", encoding="utf-8") as f:
        for name in include_for_run_test:
            f.write(f"{name}\n")

    with open(args.output_exclude, "w", encoding="utf-8") as f:
        for name in exclude_for_run_test:
            f.write(f"{name}\n")

    with open(args.output_extra, "w", encoding="utf-8") as f:
        for path in extra_for_pytest:
            f.write(f"{path}\n")

    print(f"[resolve_whitelist] Written {len(include_for_run_test)} tests to {args.output_include}")
    print(f"[resolve_whitelist] Written {len(exclude_for_run_test)} tests to {args.output_exclude}")
    print(f"[resolve_whitelist] Written {len(extra_for_pytest)} tests to {args.output_extra}")

    if args.verbose:
        print("\n=== Include tests for run_test.py ===")
        for name in include_for_run_test[:20]:
            print(f"  {name}")
        if len(include_for_run_test) > 20:
            print(f"  ... and {len(include_for_run_test) - 20} more")

        print("\n=== Exclude tests for run_test.py ===")
        for name in exclude_for_run_test[:20]:
            print(f"  {name}")
        if len(exclude_for_run_test) > 20:
            print(f"  ... and {len(exclude_for_run_test) - 20} more")

        print("\n=== Extra pytest tests ===")
        for path in extra_for_pytest[:20]:
            print(f"  {path}")
        if len(extra_for_pytest) > 20:
            print(f"  ... and {len(extra_for_pytest) - 20} more")


if __name__ == "__main__":
    main()