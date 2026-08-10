#!/usr/bin/env python3
"""
Shard test files by business category using round-robin distribution.

Scans all test_*.py files under test_dir, classifies each file into a
category based on the classification rules in the config YAML, then
distributes files across shards using round-robin.

Classification priority (3 passes):
    1. files:  exact file match across all categories (first match wins)
    2. paths:  directory prefix match for remaining files
    3. others: catch-all for any test_*.py not matched above

Usage:
    python shard_test_files.py \
        --test-dir /path/to/pytorch/test \
        --categories-config .github/config/nightly_v3_test_whitelist.yml \
        --output-dir /path/to/output
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

try:
    import yaml
except ImportError:
    yaml = None


# ==============================================================================
# Config Loading
# ==============================================================================


def load_categories_config(config_path: str) -> Dict:
    """Load categories config YAML and return the full config dict.

    Returns a dict with:
        'exclude':    list of paths to exclude from scanning
        'categories': dict of category_name -> category config
    """
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Categories config not found: {config_file}")

    raw_text = config_file.read_text(encoding="utf-8")

    if yaml is not None:
        data = yaml.safe_load(raw_text) or {}
    else:
        data = _parse_simple_categories_yaml(raw_text)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML object in {config_file}, got {type(data).__name__}")

    return {
        "exclude": data.get("exclude", []),
        "categories": data.get("categories", {}),
    }


def _parse_simple_categories_yaml(raw_text: str) -> Dict:
    """Parse categories YAML without yaml library (minimal parser).

    Supports top-level 'exclude' list and 'paths'/'files' list fields
    under each category. Handles 0/2/4/6-space indents.
    """
    result = {"categories": {}, "exclude": []}
    current_category = None
    current_list_key = None
    in_exclude = False

    for raw_line in raw_text.splitlines():
        without_comment = raw_line.split("#", 1)[0].rstrip()
        if not without_comment.strip():
            continue

        stripped = without_comment.lstrip()
        indent = len(without_comment) - len(stripped)

        # Top-level key (indent 0)
        if indent == 0 and stripped.endswith(":"):
            key = stripped[:-1].strip()
            in_exclude = (key == "exclude")
            current_category = None
            current_list_key = None
            continue

        # Top-level list items (indent 2) — for exclude
        if indent == 2 and in_exclude and stripped.startswith("- "):
            value = stripped[2:].strip().strip("\"'")
            if value:
                result["exclude"].append(value)
            continue

        # Category name (indent 2)
        if indent == 2 and stripped.endswith(":"):
            current_category = stripped[:-1].strip()
            result["categories"][current_category] = {}
            current_list_key = None
            in_exclude = False
            continue

        # Category key-value (indent 4)
        if current_category and indent == 4:
            current_list_key = None
            if ":" in stripped:
                key, val = stripped.split(":", 1)
                key = key.strip()
                val = val.strip()
                cat = result["categories"][current_category]
                if key in ("files", "paths"):
                    current_list_key = key
                    cat[key] = []
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                    cat[key] = val
            continue

        # Category list items (indent 6)
        if current_category and indent == 6 and current_list_key:
            if stripped.startswith("- "):
                value = stripped[2:].strip().strip("\"'")
                if value:
                    cat = result["categories"][current_category]
                    cat[current_list_key].append(value)
            continue

    return result


# ==============================================================================
# File Scanning
# ==============================================================================


def scan_all_test_files(test_dir: Path) -> Set[str]:
    """Recursively scan test_dir for executable test_*.py files.

    Only files that contain a test entry point (``run_tests()`` or
    ``unittest.main()``) are returned.  Sub-files that are imported by
    a parent file (e.g. ``jit/test_tracer.py`` imported by
    ``test_jit.py``) are skipped — they call ``raise_on_run_directly()``
    or ``raise RuntimeError(...)`` in their ``__main__`` block and would
    fail if executed directly.

    Returns a set of relative paths prefixed with 'test/', e.g.
    'test/nn/test_convolution.py'.
    """
    all_files = set()
    for path in test_dir.rglob("test_*.py"):
        if path.is_file() and _is_executable_test_file(path):
            rel = path.relative_to(test_dir.parent)
            all_files.add(str(rel))
    return all_files


def _is_executable_test_file(path: Path) -> bool:
    """Check whether a test file can be executed directly.

    Executable files call ``run_tests()`` or ``unittest.main()`` in
    their ``__main__`` block.  Non-executable files (sub-modules,
    helpers, disabled tests) raise ``RuntimeError`` or have no
    ``__main__`` block at all.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "run_tests(" in content or "unittest.main(" in content


# ==============================================================================
# Classification
# ==============================================================================


def classify_files(
    all_files: Set[str],
    categories: Dict,
    exclude: List[str] = None,
) -> Dict[str, List[str]]:
    """Classify files into categories using 3-pass priority matching.

    Excluded paths are removed from the file set before classification,
    so they never enter any category (not even others).

    Pass 1 — files: exact file match across all categories (first match wins)
    Pass 2 — paths: directory prefix match for remaining files
    Pass 3 — others: catch-all for unclassified files

    Returns dict mapping category_name -> sorted list of file paths.
    """
    classified: Dict[str, List[str]] = {name: [] for name in categories}
    working_set = set(all_files)

    # Pass 0: remove excluded paths
    if exclude:
        excluded_count = 0
        for f in list(working_set):
            for excl_path in exclude:
                prefix = excl_path if excl_path.endswith("/") else excl_path + "/"
                if f.startswith(prefix):
                    working_set.discard(f)
                    excluded_count += 1
                    break
        if excluded_count:
            print(f"  Excluded {excluded_count} files matching exclude paths: {exclude}")

    unclassified = working_set

    # Pass 1: exact file match (first category wins)
    for cat_name, cat_config in categories.items():
        exact_files = cat_config.get("files", [])
        for f in exact_files:
            if f in unclassified:
                classified[cat_name].append(f)
                unclassified.discard(f)

    # Pass 2: directory prefix match
    for cat_name, cat_config in categories.items():
        paths = cat_config.get("paths", [])
        if not paths:
            continue
        remaining = list(unclassified)
        for f in remaining:
            for dir_path in paths:
                prefix = dir_path if dir_path.endswith("/") else dir_path + "/"
                if f.startswith(prefix):
                    classified[cat_name].append(f)
                    unclassified.discard(f)
                    break

    # Pass 3: catch-all into 'others' category
    if "others" in classified and unclassified:
        classified["others"].extend(sorted(unclassified))
        unclassified.clear()
    elif unclassified:
        print(
            f"  WARNING: {len(unclassified)} files unclassified (no 'others' category):",
            file=sys.stderr,
        )
        for f in sorted(unclassified)[:10]:
            print(f"    {f}", file=sys.stderr)
        if len(unclassified) > 10:
            print(f"    ... and {len(unclassified) - 10} more", file=sys.stderr)

    # Sort each category's file list
    for cat_name in classified:
        classified[cat_name].sort()

    return classified


# ==============================================================================
# Sharding
# ==============================================================================


def split_round_robin(files: List[str], num_shards: int) -> List[List[str]]:
    """Round-robin distribute files across shards for balanced load."""
    if num_shards <= 0:
        num_shards = 1
    shards = [[] for _ in range(num_shards)]
    for i, f in enumerate(sorted(files)):
        shards[i % num_shards].append(f)
    return shards


def save_shard_json(
    output_dir: Path,
    category: str,
    shard_num: int,
    num_shards: int,
    files: List[str],
) -> Path:
    """Save {category}_files_shard_{n}.json."""
    data = {
        "shard": shard_num,
        "num_shards": num_shards,
        "test_type": category,
        "total_files": len(files),
        "files": files,
    }
    path = output_dir / f"{category}_files_shard_{shard_num}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ==============================================================================
# Main
# ==============================================================================


def main():
    args = parse_args()

    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_categories_config(args.categories_config)
    exclude = config.get("exclude", [])
    categories = config.get("categories", {})

    print("=" * 80)
    print("Sharding test files by business category (round-robin)")
    print("=" * 80)
    print(f"Test directory: {test_dir}")
    print(f"Categories: {len(categories)}")
    if exclude:
        print(f"Exclude paths: {exclude}")
    print()

    # Step 1: scan all test files
    all_files = scan_all_test_files(test_dir)
    print(f"Scanned {len(all_files)} test_*.py files under {test_dir}")
    print()

    # Step 2: classify files into categories
    classified = classify_files(all_files, categories, exclude)

    # Step 3: shard each category
    summary_categories = {}
    total_files_all = 0

    for cat_name, cat_config in categories.items():
        config_shards = cat_config.get("shards", 1)
        runner = cat_config.get("runner", "linux-aarch64-a3-8")
        devices_per_proc = cat_config.get("devices_per_proc", 1)
        # Derive npu_count from runner label suffix (same as v2):
        #   "linux-aarch64-a3-8"  → 8
        #   "linux-aarch64-b2-16" → 16
        npu_count = int(runner.rsplit("-", 1)[-1]) if runner else 8
        cat_files = classified.get(cat_name, [])

        # Print classification details
        paths = cat_config.get("paths", [])
        files = cat_config.get("files", [])
        print(f"--- Category: {cat_name} ---")
        if paths:
            print(f"  Paths: {paths}")
        if files:
            print(f"  Configured files: {len(files)}")
        print(f"  Matched files: {len(cat_files)}")
        print(f"  Shards: {config_shards}")
        print(f"  Runner: {runner}  (npu_count={npu_count})")
        print(f"  Devices per proc: {devices_per_proc}  "
              f"(concurrency: {max(1, npu_count // devices_per_proc)} workers)")

        if not cat_files:
            print(f"  WARNING: No files matched for category '{cat_name}'")
            summary_categories[cat_name] = {
                "num_shards": config_shards,
                "total_files": 0,
                "shard_sizes": [],
                "runner": runner,
                "npu_count": npu_count,
                "devices_per_proc": devices_per_proc,
            }
            print()
            continue

        shards = split_round_robin(cat_files, config_shards)
        num_shards = config_shards

        shard_sizes = []
        for i, shard_files in enumerate(shards, 1):
            save_shard_json(output_dir, cat_name, i, num_shards, shard_files)
            shard_sizes.append(len(shard_files))
            print(f"  Shard {i}/{num_shards}: {len(shard_files)} files")

        total_files_all += len(cat_files)
        summary_categories[cat_name] = {
            "num_shards": num_shards,
            "total_files": len(cat_files),
            "shard_sizes": shard_sizes,
            "runner": runner,
            "npu_count": npu_count,
            "devices_per_proc": devices_per_proc,
        }
        print()

    # Save summary
    summary = {
        "categories": summary_categories,
        "total_cases": None,
        "total_files_scanned": len(all_files),
    }
    summary_file = output_dir / "cases_collection_summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Summary saved to {summary_file}")

    print()
    print("=" * 80)
    print("Sharding Complete")
    print("=" * 80)
    for cat_name, cat_summary in summary_categories.items():
        sizes = cat_summary["shard_sizes"]
        print(
            f"  {cat_name}: {cat_summary['total_files']} files -> "
            f"{cat_summary['num_shards']} shards (sizes: {sizes})"
        )
    print(f"  Total: {total_files_all} files (scanned: {len(all_files)})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shard test files by business category"
    )
    parser.add_argument("--test-dir", required=True, help="PyTorch test directory")
    parser.add_argument(
        "--categories-config", required=True, help="Path to categories config YAML"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for shard JSONs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
