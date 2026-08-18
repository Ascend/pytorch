#!/usr/bin/env python3
"""
Collect all test cases and split into shards.

This script runs in prepare job (once) to:
1. Discover test files by type (distributed/regular)
2. Collect all test cases via pytest --collect-only
3. Split cases evenly into N shards
4. Output shard JSON files for each type
5. Save collection error logs for failed files

Usage:
    python collect_all_cases.py \
        --test-dir /path/to/pytorch/test \
        --hw-classification ACCELERATOR \
        --distributed-shards 2 \
        --regular-shards 5 \
        --output-dir /path/to/output \
        --error-log-dir /path/to/error_logs \
        --parallel 16
"""

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None



# ==============================================================================
# Category Configuration Loading
# ==============================================================================


def load_categories_config(config_path: Optional[str]) -> Tuple[Dict[str, Dict], List[str]]:
    """Load category-driven configuration from YAML.

    Supports two formats:

    **New format** (category-driven):
        exclude:
          - test/cpython
          - test/quantization/core/experimental
        categories:
          core:
            workers: 32
            execution: concurrent
            files: [...]
          distributed:
            workers: 1
            execution: serial
            files: [...]

    **Legacy format** (flat whitelist, backward compatible):
        whitelist: [...]
        blacklist: []

    For the legacy format, files are split into "distributed" (paths
    starting with ``test/distributed/``) and "regular" (everything else)
    to preserve existing behaviour.

    Returns:
        Tuple of (categories_dict, exclude_list).
        categories_dict maps category name to {files, paths, workers, execution}.
        exclude_list is a list of directory/file paths to skip entirely.
    """
    if not config_path:
        raise ValueError("No config path provided; cannot load categories.")

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    raw = p.read_text(encoding="utf-8")

    if yaml is not None:
        data = yaml.safe_load(raw) or {}
    else:
        raise RuntimeError("PyYAML is required for category config parsing.")

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object, got {type(data).__name__}")

    # New format: categories key present
    if "categories" in data:
        exclude_list = []
        raw_exclude = data.get("exclude", [])
        if isinstance(raw_exclude, list):
            exclude_list = [str(e).rstrip("/") for e in raw_exclude if isinstance(e, str) and e.strip()]
        elif raw_exclude:
            print(f"  WARNING: 'exclude' must be a list, got {type(raw_exclude).__name__}; ignoring", file=sys.stderr)

        result = {}
        for cat_name, cat_cfg in data["categories"].items():
            if not isinstance(cat_cfg, dict):
                raise ValueError(
                    f"Category '{cat_name}' must be a dict, got {type(cat_cfg).__name__}"
                )
            files = cat_cfg.get("files", [])
            if not isinstance(files, list):
                raise ValueError(
                    f"Category '{cat_name}' files must be a list, got {type(files).__name__}"
                )
            result[cat_name] = {
                "files": list(dict.fromkeys(files)),  # deduplicate
                "paths": cat_cfg.get("paths", []),
                "workers": int(cat_cfg.get("workers", 32)),
                "execution": cat_cfg.get("execution", "concurrent"),
                "runner": cat_cfg.get("runner", "linux-aarch64-a3-8"),
            }
            if not isinstance(result[cat_name]["paths"], list):
                raise ValueError(
                    f"Category '{cat_name}' paths must be a list, got "
                    f"{type(result[cat_name]['paths']).__name__}"
                )
        return result, exclude_list

    # Legacy format: flat whitelist
    if "whitelist" in data:
        whitelist = data.get("whitelist", [])
        if not isinstance(whitelist, list):
            raise ValueError(f"Expected 'whitelist' to be a list")
        dist_files = [f for f in whitelist if f.startswith("test/distributed/")]
        reg_files = [f for f in whitelist if not f.startswith("test/distributed/")]
        result = {}
        if dist_files:
            result["distributed"] = {
                "files": dist_files,
                "workers": 1,
                "execution": "serial",
            }
        if reg_files:
            result["regular"] = {
                "files": reg_files,
                "workers": 32,
                "execution": "concurrent",
            }
        return result, []

    raise ValueError(
        f"Unknown config format in {p}: expected 'categories' or 'whitelist' key"
    )


def classify_files_full_scan(
    all_files: List[str],
    categories: Dict[str, Dict],
    exclude: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Classify scanned files into categories using 3-pass first-match-wins.

    Ported from v3/shard_test_files.py. Used when --full-scan is active:
    the config's files/paths act as classification rules, not a whitelist.

    Excluded files (from top-level ``exclude`` config key) are removed
    before classification begins.  Each exclude entry is matched as:
      - Directory prefix: ``test/cpython`` matches ``test/cpython/...``
      - Exact file:       ``test/foo/test_bar.py`` matches only that file

    Pass 1 — files:  exact file match across all categories (first match wins)
    Pass 2 — paths:  directory prefix match for remaining files
    Pass 3 — others: catch-all for any unmatched test_*.py

    Args:
        all_files: List of test file paths (e.g. ["test/nn/test_foo.py", ...])
        categories: Dict from load_categories_config, each with files/paths.
        exclude: Optional list of directory/file paths to skip entirely.

    Returns:
        Dict mapping category name -> sorted list of file paths.
    """
    classified: Dict[str, List[str]] = {name: [] for name in categories}
    working_set = set(all_files)

    # Pre-pass: remove excluded files
    if exclude:
        excluded_count = 0
        for pattern in exclude:
            prefix = pattern.rstrip("/") + "/"
            to_remove = {f for f in working_set if f.startswith(prefix) or f == pattern}
            working_set -= to_remove
            excluded_count += len(to_remove)
        if excluded_count:
            print(f"  [exclude] Removed {excluded_count} files matching {len(exclude)} exclude patterns")

    # Pass 1: exact file match (first category wins)
    for cat_name, cat_cfg in categories.items():
        for f in cat_cfg.get("files", []):
            if f in working_set:
                classified[cat_name].append(f)
                working_set.discard(f)

    # Pass 2: directory prefix match
    for cat_name, cat_cfg in categories.items():
        for dir_path in cat_cfg.get("paths", []):
            prefix = dir_path.rstrip("/") + "/"
            remaining = list(working_set)
            for f in remaining:
                if f.startswith(prefix):
                    classified[cat_name].append(f)
                    working_set.discard(f)

    # Pass 3: catch-all into 'others' category
    if "others" in classified and working_set:
        classified["others"].extend(sorted(working_set))
        working_set.clear()
    elif working_set:
        print(f"  WARNING: {len(working_set)} files unclassified "
              f"(no 'others' category in config)", file=sys.stderr)
        for f in sorted(working_set)[:10]:
            print(f"    {f}", file=sys.stderr)
        if len(working_set) > 10:
            print(f"    ... and {len(working_set) - 10} more", file=sys.stderr)

    for cat_name in classified:
        classified[cat_name].sort()

    return classified


# ==============================================================================
# Skip List Loading & Filtering
# ==============================================================================


def load_skip_list(skip_list_path: Optional[str]) -> set:
    """Load skip list and return a set of nodeids to skip.

    Supports three formats:
      - JSONL (.jsonl): line 1 is {"_meta": {...}}, subsequent lines are
        {"nodeid": "...", "reason": "..."}.  Only the ``nodeid`` value is
        consumed; ``reason`` is for human review and discarded at load time.
      - JSON object: {"version": 1, "skip_nodeids": ["nodeid1", ...]}
      - JSON array: ["nodeid1", ...]

    Returns empty set if path is None, file not found, or empty.
    Never raises — all errors fall back to empty set with a warning so
    that collection proceeds normally (backward compatible).
    """
    if not skip_list_path:
        return set()

    p = Path(skip_list_path)
    if not p.exists():
        print(f"  WARNING: skip list file not found: {p}, skipping filter")
        return set()

    if p.suffix == ".jsonl":
        return _load_skip_list_jsonl(p)

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: Failed to load skip list {p}: {e}")
        return set()

    if isinstance(data, dict):
        nodeids = data.get("skip_nodeids", [])
    elif isinstance(data, list):
        nodeids = data
    else:
        print(f"  WARNING: skip list JSON is neither object nor array: {p}")
        return set()

    skip_set = set(n for n in nodeids if isinstance(n, str) and n)
    print(f"  Loaded skip list: {len(skip_set)} nodeids from {p}")
    return skip_set


def _load_skip_list_jsonl(p: Path) -> set:
    """Load a JSONL skip list (one JSON object per line).

    Line 1 is expected to be a ``{"_meta": {...}}`` metadata record and is
    skipped.  Every subsequent line must be a JSON object with at least a
    ``nodeid`` key; the ``reason`` key (if present) is ignored.
    """
    skip_set: set = set()
    meta_seen = False
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and "_meta" in obj:
                    meta_seen = True
                    continue
                nodeid = obj.get("nodeid") if isinstance(obj, dict) else None
                if isinstance(nodeid, str) and nodeid:
                    skip_set.add(nodeid)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: Failed to load JSONL skip list {p}: {e}")
        return set()

    print(f"  Loaded skip list: {len(skip_set)} nodeids from {p}"
          f"{' (meta line found)' if meta_seen else ''}")
    return skip_set


def filter_skipped_cases(cases: List[Dict], skip_nodeids: set) -> List[Dict]:
    """Remove cases whose nodeid matches the skip set.

    Prints before/after counts. If skip_nodeids is empty, returns cases
    unchanged (zero overhead, backward compatible).
    """
    if not skip_nodeids:
        return cases

    original_count = len(cases)
    filtered = [c for c in cases if c.get("nodeid", "") not in skip_nodeids]
    skipped_count = original_count - len(filtered)
    print(f"  Skip list filter: {original_count} -> {len(filtered)} cases "
          f"(removed {skipped_count})")
    return filtered


def _normalize_test_file_path(test_file: str) -> str:
    """
    Remove 'test/' prefix from test file path if present.

    Args:
        test_file: Test file path (e.g., "test/distributed/pipelining/test_backward.py")

    Returns:
        Relative path without 'test/' prefix
    """
    if test_file.startswith("test/"):
        return test_file[5:]
    return test_file


def get_test_file_parent_dir(test_file: str, test_dir: Path) -> Path:
    """
    Get the parent directory of a test file.

    This directory should be added to PYTHONPATH to enable
    imports of sibling modules (e.g., model_registry.py).

    Args:
        test_file: Test file path (e.g., "test/distributed/pipelining/test_backward.py")
        test_dir: Path to PyTorch test directory

    Returns:
        Path to the test file's parent directory
    """
    test_file_rel = _normalize_test_file_path(test_file)
    test_file_path = Path(test_file_rel)
    return test_dir / test_file_path.parent


def collect_cases_for_file(
    test_file: str,
    test_dir: Path,
    hw_classification: Optional[List[str]] = None,
) -> Tuple[str, str, List[str], bool, str]:
    """
    Collect test cases from a single file.

    Adds test file's parent directory to PYTHONPATH to enable
    imports of sibling modules (e.g., 'from model_registry import MLPModule').

    Args:
        test_file: Test file path (e.g., "test/distributed/test_c10d.py")
        test_dir: Path to PyTorch test directory
        hw_classification: Optional list of hardware classification filters
            (e.g., ["ACCELERATOR"]). When set, --hw-classification is passed
            to pytest --collect-only so only tests with matching hw_classification
            class attributes are collected. Files with no matching tests return
            exit code 5, which is treated as success (0 cases) in this mode.

    Returns:
        Tuple of (test_file, display_name, nodeids, success, error_message)
        - test_file: Original test file path
        - display_name: Short name for logging (remove test/ prefix and .py suffix)
        - nodeids: List of collected test case nodeids
        - success: True if collection succeeded without errors
        - error_message: Error details if collection failed, empty string otherwise
    """
    test_file_rel = _normalize_test_file_path(test_file)

    # Extract display name (remove .py suffix)
    display_name = test_file_rel
    if display_name.endswith(".py"):
        display_name = display_name[:-3]

    # Get test file's parent directory for PYTHONPATH
    test_file_dir = get_test_file_parent_dir(test_file, test_dir)

    # Build environment with test file directory in PYTHONPATH
    env = os.environ.copy()
    env["PYTORCH_TESTING_DEVICE_ONLY_FOR"] = "privateuse1"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(test_file_dir) + (":" + existing_pythonpath if existing_pythonpath else "")

    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "--quiet",
        test_file_rel,
    ]
    # --hw-classification must come AFTER the test file path because
    # conftest.py defines it with nargs="+" (greedy), which would consume
    # the file path as a classification value if placed before it.
    if hw_classification:
        command.append("--hw-classification")
        command.extend(hw_classification)

    print(f"  [{display_name}] Collecting: {' '.join(command)}", flush=True)

    try:
        result = subprocess.run(
            command,
            cwd=str(test_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )

        nodeids = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            # pytest --collect-only -q outputs clean nodeids, one per line
            # Filter rules:
            # 1. Skip empty lines
            # 2. Skip summary lines (contain "collected" or "selected")
            # 3. Skip separator lines (start with "=")
            # 4. Must contain ".py::" to ensure it's a Python test file nodeid
            if not stripped:
                continue
            if "collected" in stripped or "selected" in stripped:
                continue
            if stripped.startswith("="):
                continue
            if ".py::" in stripped:
                nodeids.append(stripped)

        # Check for collection errors based on pytest exit codes:
        #   0: all passed (success)
        #   2: pytest error (includes collection errors like ImportError)
        #   3: all skipped (success)
        #   4: command line error (error)
        #   5: no tests collected
        # When hw_classification or device filtering (PYTORCH_TESTING_DEVICE_ONLY_FOR)
        # is active, exit code 5 is expected for files that have no test classes
        # matching the requested device/classification — this is normal because
        # many files only contain CPU tests or are not yet annotated.
        # Without any filtering, exit code 5 means a selected file has 0
        # cases, which indicates a problem.
        device_filtered = bool(env.get("PYTORCH_TESTING_DEVICE_ONLY_FOR"))
        if result.returncode in (0, 3):
            return (test_file, display_name, nodeids, True, "")
        elif (hw_classification or device_filtered) and result.returncode == 5:
            return (test_file, display_name, nodeids, True, "")
        else:
            # returncode 2, 4, 5: real collection error
            # returncode 5 specifically means no tests collected - a problem for selected files
            error_msg = result.stdout.strip()
            if result.stderr.strip():
                error_msg += "\n--- stderr ---\n" + result.stderr.strip()

            # Diagnostic info for first failure: capture env state
            diag_lines = []
            try:
                import subprocess as sp
                diag_lines.append("--- Diagnostics ---")
                diag_lines.append("LD_LIBRARY_PATH: " + os.environ.get("LD_LIBRARY_PATH", "NOT SET"))
                diag_lines.append("PATH: " + os.environ.get("PATH", "NOT SET"))
                r = sp.run(["find", "/usr/local/Ascend", "-name", "libhccl.so"], capture_output=True, text=True, timeout=10)
                diag_lines.append("find libhccl.so: " + (r.stdout.strip() or "NOT FOUND"))
                r2 = sp.run(["cat", "/usr/local/Ascend/cann/version.cfg"], capture_output=True, text=True, timeout=5)
                diag_lines.append("CANN version: " + (r2.stdout.strip() or "MISSING"))
                r3 = sp.run(["python3", "-c", "import torch; print('torch:', torch.__version__)"], capture_output=True, text=True, timeout=10, env=os.environ, cwd="/tmp")
                diag_lines.append("torch version: " + (r3.stdout.strip() or r3.stderr.strip()))
            except Exception:
                diag_lines.append("--- Diagnostics FAILED ---")
            error_msg += "\n" + "\n".join(diag_lines)

            return (test_file, display_name, nodeids, False, error_msg)

    except subprocess.TimeoutExpired:
        error_msg = f"TIMEOUT: Collection took >120s for {display_name}"
        return (test_file, display_name, [], False, error_msg)
    except Exception as e:
        error_msg = f"ERROR: {e}"
        return (test_file, display_name, [], False, error_msg)


def collect_all_cases(
    test_files: List[str],
    test_dir: Path,
    error_log_dir: Path,
    parallel: int = 16,
    hw_classification: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Collect all cases from all files.

    Args:
        test_files: List of test file paths
        test_dir: Path to PyTorch test directory
        error_log_dir: Directory to save error logs for failed collections
        parallel: Number of parallel workers
        hw_classification: Optional hardware classification filter
            (e.g., ["ACCELERATOR"])

    Returns:
        List of dicts with nodeid and file for each collected case
    """
    all_cases = []
    failed_files = []  # Track files with collection errors for logging

    if hw_classification:
        print(f"Collecting cases from {len(test_files)} files with {parallel} workers "
              f"(hw_classification={hw_classification})...")
    else:
        print(f"Collecting cases from {len(test_files)} files with {parallel} workers...")
    print("=" * 60)

    # Create error log directory
    error_log_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(collect_cases_for_file, f, test_dir, hw_classification): f
            for f in test_files
        }

        completed = 0
        successful_count = 0
        failed_count = 0
        total_cases = 0

        for future in as_completed(futures):
            test_file, display_name, nodeids, success, error_msg = future.result()
            completed += 1

            if success:
                successful_count += 1
                # Print concise log for successful files
                print(f"  {display_name}: {len(nodeids)} cases")
                for nodeid in nodeids:
                    all_cases.append({
                        "nodeid": nodeid,
                        "file": test_file,
                    })
            else:
                failed_count += 1
                # Print concise log for failed files
                print(f"  [FAILED] {display_name}: {len(nodeids)} cases")
                # Save error details to log file
                failed_files.append({
                    "file": display_name,
                    "error": error_msg,
                    "cases": len(nodeids),
                    "test_file": test_file,
                })
                # Still add any cases that were collected despite errors
                for nodeid in nodeids:
                    all_cases.append({
                        "nodeid": nodeid,
                        "file": test_file,
                    })

            # Update total cases count for progress display
            total_cases += len(nodeids)

            # Print progress summary every 100 files
            if completed % 100 == 0:
                print(f"  [Progress: {completed}/{len(test_files)} files, {successful_count} ok, {failed_count} failed, {total_cases} cases]")

    print("=" * 60)

    # Save error logs to files
    if failed_files:
        save_error_logs(failed_files, error_log_dir)

    # Final summary
    print(f"Collection complete: {len(all_cases)} cases from {successful_count}/{len(test_files)} files")
    if failed_count > 0:
        print(f"  WARNING: {failed_count} files had collection errors (logs saved to {error_log_dir})")

    return all_cases


def save_error_logs(failed_files: List[Dict], error_log_dir: Path) -> None:
    """
    Save collection error logs to individual files and create a summary.

    Args:
        failed_files: List of dicts with file, error, cases info
        error_log_dir: Directory to save error logs
    """
    print(f"Saving error logs for {len(failed_files)} failed files...")

    # Save individual error log files
    for failed in failed_files:
        # Create safe filename from display name (replace / with _)
        safe_name = failed['file'].replace('/', '_')
        log_file = error_log_dir / f"{safe_name}.log"

        # Write error log
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"File: {failed['file']}\n")
            f.write(f"Cases collected: {failed['cases']}\n")
            f.write(f"Test file path: {failed['test_file']}\n")
            f.write("=" * 80 + "\n")
            f.write("Collection Error:\n")
            f.write("=" * 80 + "\n")
            f.write(failed['error'])
            f.write("\n")

    # Save summary JSON
    summary_file = error_log_dir / "collection_errors_summary.json"
    summary_data = {
        "total_failed": len(failed_files),
        "failed_files": [
            {
                "file": f['file'],
                "cases": f['cases'],
                "test_file": f['test_file'],
                "log_file": f"{f['file'].replace('/', '_')}.log",
            }
            for f in failed_files
        ],
    }
    summary_file.write_text(json.dumps(summary_data, indent=2), encoding='utf-8')

    print(f"  Error logs saved to {error_log_dir}")
    print(f"  Summary: {summary_file}")


def split_cases_into_shards(cases: List[Dict], num_shards: int) -> List[List[Dict]]:
    """Split cases evenly into shards."""
    total = len(cases)
    base_size = total // num_shards
    remainder = total % num_shards

    shards = []
    start = 0
    for i in range(num_shards):
        size = base_size + (1 if i < remainder else 0)
        shards.append(cases[start:start + size])
        start += size

    return shards


def save_cases_by_file(
    cases: List[Dict],
    test_files: List[str],
    test_type: str,
    output_dir: Path,
) -> Dict:
    """
    Save cases grouped by file in JSONL format.

    Includes all test files, even those with 0 cases collected.

    Output format (JSONL, one JSON object per line):
    Line 1: {"total_file":<count>,"total_cases":<count>}
    Line 2+: {"file_path":"...","case_count":<count>,"cases":["nodeid1","nodeid2",...]}
    """
    # Group cases by file
    file_groups: Dict[str, List[str]] = {}
    for case in cases:
        file_path = case["file"]
        if file_path not in file_groups:
            file_groups[file_path] = []
        file_groups[file_path].append(case["nodeid"])

    output_file = output_dir / f"{test_type}_cases_by_file.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Line 1: summary
        summary_line = json.dumps({
            "total_file": len(test_files),
            "total_cases": len(cases),
        }, separators=(',', ':'))
        f.write(summary_line + '\n')

        # Line 2+: file data (sorted by file path)
        for file_path in sorted(test_files):
            nodeids = file_groups.get(file_path, [])
            file_line = json.dumps({
                "file_path": file_path,
                "case_count": len(nodeids),
                "cases": nodeids,
            }, separators=(',', ':'))
            f.write(file_line + '\n')

    print(f"  Cases by file (JSONL): {len(test_files)} files -> {output_file}")

    return {
        "test_type": test_type,
        "total_files": len(test_files),
        "total_cases": len(cases),
    }


def save_shards(
    cases: List[Dict],
    num_shards: int,
    test_type: str,
    output_dir: Path,
) -> Dict:
    """Save shard JSONs and return summary.

    When num_shards is 0 (no cases), no shard files are written.
    """
    if num_shards == 0:
        return {
            "test_type": test_type,
            "num_shards": 0,
            "total_cases": len(cases),
            "shard_sizes": [],
        }

    shards = split_cases_into_shards(cases, num_shards)

    print(f"\nSaving {test_type} shards...")
    for i, shard_cases in enumerate(shards, 1):
        shard_file = output_dir / f"{test_type}_cases_shard_{i}.json"
        shard_data = {
            "shard": i,
            "num_shards": num_shards,
            "test_type": test_type,
            "total_cases": len(shard_cases),
            "cases": shard_cases,
        }
        shard_file.write_text(json.dumps(shard_data, indent=2), encoding="utf-8")
        print(f"  Shard {i}: {len(shard_cases)} cases -> {shard_file}")

    return {
        "test_type": test_type,
        "num_shards": num_shards,
        "total_cases": len(cases),
        "shard_sizes": [len(s) for s in shards],
    }


def main():
    args = parse_args()

    test_dir = Path(args.test_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Error log directory for failed collections
    error_log_dir = Path(args.error_log_dir).resolve() if args.error_log_dir else output_dir / "collection_errors"
    error_log_dir.mkdir(parents=True, exist_ok=True)

    hw_classification = args.hw_classification if args.hw_classification else None
    case_paths_config = args.case_paths_config if args.case_paths_config else None

    # Load skip list once (reused for all categories)
    skip_set = load_skip_list(args.skip_list)

    # Load categories from config (supports new "categories:" and legacy "whitelist:" formats)
    if case_paths_config:
        categories, exclude_list = load_categories_config(case_paths_config)
    else:
        # No config: fall back to scanning all test_*.py files via discover_test_files
        import discover_test_files
        all_files, _ = discover_test_files.discover_test_files(test_dir, "regular", None)
        categories = {"regular": {"files": all_files, "workers": 32, "execution": "concurrent"}}
        exclude_list = []

    # Full-scan mode: scan ALL test_*.py and use config as categorization mapping
    if args.full_scan:
        import discover_test_files
        all_scanned = discover_test_files.discover_raw_test_files(test_dir)

        classified = classify_files_full_scan(all_scanned, categories, exclude=exclude_list)

        # Backfill each category's files with classified results
        for cat_name in categories:
            categories[cat_name]["files"] = classified.get(cat_name, [])

        print(f"[full-scan] Scanned {len(all_scanned)} test_*.py files, "
              f"classified into {len(categories)} categories via config + auto-routing")
        if exclude_list:
            print(f"  [exclude] {len(exclude_list)} exclude patterns: {exclude_list}")

    print("Categories loaded:")
    for cat_name, cat_cfg in categories.items():
        print(f"  {cat_name}: {len(cat_cfg['files'])} files, workers={cat_cfg['workers']}, "
              f"execution={cat_cfg['execution']}")

    # Determine thresholds
    regular_threshold = getattr(args, 'regular_threshold', 10000)
    distributed_threshold = getattr(args, 'distributed_threshold', 1000)

    summary_categories = {}
    total_cases = 0
    total_files = 0

    for cat_name, cat_config in categories.items():
        print("\n" + "=" * 80)
        print(f"Collecting {cat_name} test cases")
        print("=" * 80)

        files = cat_config["files"]

        # Defensive filter: only collect test_*.py files
        files = [f for f in files if Path(f).name.startswith("test_") and f.endswith(".py")]
        print(f"Files for {cat_name}: {len(files)} (after test_*.py filter)")

        if not files:
            print(f"  No test files for category '{cat_name}', skipping.")
            summary_categories[cat_name] = {
                "test_type": cat_name,
                "num_shards": 0,
                "total_cases": 0,
                "total_files": len(files),
                "workers": cat_config.get("workers", 32),
                "execution": cat_config.get("execution", "concurrent"),
                "shard_sizes": [],
            }
            continue

        cases = collect_all_cases(
            files, test_dir, error_log_dir / cat_name,
            args.parallel, hw_classification,
        )
        print(f"Total {cat_name} cases: {len(cases)}")

        cases = filter_skipped_cases(cases, skip_set)

        cases.sort(key=lambda c: (c.get("file", ""), c.get("nodeid", "")))

        # Threshold-based shard count
        threshold = distributed_threshold if cat_name == "distributed" else regular_threshold
        if len(cases) > 0:
            num_shards = max(1, math.ceil(len(cases) / threshold))
        else:
            num_shards = 0

        print(f"  Threshold: {threshold}, Cases: {len(cases)} -> Shards: {num_shards}")

        cat_summary = save_shards(cases, num_shards, cat_name, output_dir)
        cat_summary["total_files"] = len(files)
        cat_summary["workers"] = cat_config.get("workers", 32)
        cat_summary["execution"] = cat_config.get("execution", "concurrent")
        cat_summary["runner"] = cat_config.get("runner", "linux-aarch64-a3-8")
        save_cases_by_file(cases, files, cat_name, output_dir)
        summary_categories[cat_name] = cat_summary

        total_cases += len(cases)
        total_files += len(files)

    # ========================================
    # Save overall summary
    # ========================================
    overall_summary = {
        "categories": summary_categories,
        "total_cases": total_cases,
        "total_files": total_files,
    }
    if hw_classification:
        overall_summary["hw_classification"] = hw_classification
    if case_paths_config:
        overall_summary["case_paths_config"] = case_paths_config

    summary_file = output_dir / "cases_collection_summary.json"
    summary_file.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")
    print(f"\nOverall summary saved to {summary_file}")

    # ========================================
    # Global validation
    # ========================================
    if hw_classification and total_cases == 0:
        print(f"\nERROR: --hw-classification {hw_classification} was specified but "
              f"0 cases collected from {total_files} files.")
        print("This likely means the conftest.py hw_classification plugin is not "
              "active or no test classes are annotated with the requested classification.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Collection Complete")
    print("=" * 80)
    for cat_name, cat_summary in summary_categories.items():
        n_shards = cat_summary.get("num_shards", 0)
        n_cases = cat_summary.get("total_cases", 0)
        exec_mode = categories[cat_name].get("execution", "concurrent")
        print(f"  {cat_name}: {n_cases} cases -> {n_shards} shards ({exec_mode})")
    print(f"  Total: {total_cases} cases")

def parse_args():
    parser = argparse.ArgumentParser(description="Collect and shard test cases")
    parser.add_argument("--test-dir", required=True, help="PyTorch test directory")
    parser.add_argument(
        "--case-paths-config",
        default=None,
        help="Path to whitelist/blacklist YAML (e.g., test_whitelist.yml). "
             "When set, only whitelisted files are collected; when omitted, "
             "all test_*.py files are scanned.",
    )
    parser.add_argument(
        "--hw-classification",
        nargs="+",
        default=None,
        help="Filter test cases by hardware classification (e.g., ACCELERATOR). "
             "When set, --hw-classification is passed to pytest --collect-only "
             "so only tests with matching hw_classification class attributes "
             "are collected.",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        default=False,
        help="Scan ALL test_*.py files. The case-paths-config is then used as a "
             "categorization mapping (files + paths) instead of a whitelist. "
             "Unmatched files go to 'others'; unlisted test/distributed/ files "
             "auto-route to 'distributed' via paths matching.",
    )
    parser.add_argument(
        "--distributed-shards", type=int, default=None,
        help="[DEPRECATED] Use --distributed-threshold instead. "
             "This argument is ignored when categories config is used.",
    )
    parser.add_argument(
        "--regular-shards", type=int, default=None,
        help="[DEPRECATED] Use --regular-threshold instead. "
             "This argument is ignored when categories config is used.",
    )
    parser.add_argument(
        "--regular-threshold", type=int, default=10000,
        help="Max cases per shard for non-distributed categories (default: 10000). "
             "num_shards = ceil(total_cases / threshold).",
    )
    parser.add_argument(
        "--distributed-threshold", type=int, default=1000,
        help="Max cases per shard for distributed category (default: 1000). "
             "num_shards = ceil(total_cases / threshold).",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for shard JSONs")
    parser.add_argument("--error-log-dir", help="Output directory for collection error logs (default: output-dir/collection_errors)")
    parser.add_argument("--parallel", type=int, default=16, help="Parallel collection workers")
    parser.add_argument(
        "--skip-list",
        default=None,
        help="Path to skip list JSON file. When set, matching nodeids are "
             "removed after collection and before sharding. When omitted or "
             "file not found, no filtering is applied (backward compatible).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()