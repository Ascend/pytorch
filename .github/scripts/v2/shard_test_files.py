#!/usr/bin/env python3
"""
Shard test files by business category using round-robin distribution.

Reads category config YAML, validates file existence, then distributes
files across shards using round-robin. The number of shards per category
is defined by the `shards` field in nightly_v2_test_whitelist.yml (default: 1).

Usage:
    python shard_test_files.py \
        --test-dir /path/to/pytorch/test \
        --categories-config .github/config/nightly_v2_test_whitelist.yml \
        --output-dir /path/to/output
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError:
    yaml = None


def load_categories_config(config_path: str) -> Dict:
    """Load nightly_v2_test_whitelist.yml and return the categories dict."""
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Categories config not found: {config_file}")

    raw_text = config_file.read_text(encoding="utf-8")

    if yaml is not None:
        data = yaml.safe_load(raw_text) or {}
    else:
        data = parse_simple_categories_yaml(raw_text)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML object in {config_file}, got {type(data).__name__}")

    return data.get("categories", {})


def parse_simple_categories_yaml(raw_text: str) -> Dict:
    """Parse categories YAML without yaml library (minimal parser)."""
    result = {"categories": {}}
    current_category = None
    current_field = None

    for raw_line in raw_text.splitlines():
        without_comment = raw_line.split("#", 1)[0].rstrip()
        if not without_comment.strip():
            continue

        stripped = without_comment.lstrip()
        indent = len(without_comment) - len(stripped)

        if indent == 0 and stripped.endswith(":"):
            key = stripped[:-1].strip()
            if key == "categories":
                current_field = "categories"
            continue

        if current_field == "categories":
            if indent == 2 and stripped.endswith(":"):
                current_category = stripped[:-1].strip()
                result["categories"][current_category] = {"files": []}
            elif indent == 4 and current_category:
                if stripped.startswith("- "):
                    value = stripped[2:].strip().strip("\"'")
                    if value:
                        result["categories"][current_category].setdefault("files", []).append(value)
                elif ":" in stripped:
                    key, val = stripped.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                    result["categories"][current_category][key] = val

    return result


def validate_files(files: List[str], test_dir: Path) -> Tuple[List[str], List[str]]:
    """Check files exist in test_dir, skip non-test files."""
    valid = []
    skipped = []
    for f in files:
        rel = f[5:] if f.startswith("test/") else f
        full_path = test_dir / rel
        filename = Path(rel).name
        if full_path.exists() and filename.startswith("test_") and filename.endswith(".py"):
            valid.append(f)
        else:
            skipped.append(f)
    return valid, skipped


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
    files: List,
) -> Path:
    """Save {category}_files_shard_{n}.json.

    *files* items are plain strings (whole file, shard_id=1, num_shards=1).
    """
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


def main():
    args = parse_args()

    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = load_categories_config(args.categories_config)

    print("=" * 80)
    print("Sharding test files by business category (round-robin)")
    print("=" * 80)
    print(f"Test directory: {test_dir}")
    print(f"Categories: {len(categories)}")
    print()

    summary_categories = {}
    total_files_all = 0

    for cat_name, cat_config in categories.items():
        config_shards = cat_config.get("shards", 1)
        execution = cat_config.get("execution", "concurrent")
        num_workers = cat_config.get("workers", 1 if execution == "serial" else 8)
        raw_files = cat_config.get("files", [])

        print(f"--- Category: {cat_name} ---")
        print(f"  Configured files: {len(raw_files)}")
        print(f"  Configured shards: {config_shards}")
        print(f"  Execution: {execution}")
        print(f"  Workers: {num_workers}")

        valid_files, skipped_files = validate_files(raw_files, test_dir)

        if skipped_files:
            print(f"  Skipped (not found or not test_*.py): {len(skipped_files)}")
            for sf in skipped_files:
                print(f"    - {sf}")

        print(f"  Valid files: {len(valid_files)}")

        if not valid_files:
            print(f"  WARNING: No valid files for category '{cat_name}'")
            summary_categories[cat_name] = {
                "num_shards": config_shards,
                "total_files": 0,
                "shard_sizes": [],
                "workers": num_workers,
            }
            continue

        shards = split_round_robin(valid_files, config_shards)
        num_shards = config_shards

        shard_sizes = []
        for i, shard_files in enumerate(shards, 1):
            save_shard_json(output_dir, cat_name, i, num_shards, shard_files)
            shard_sizes.append(len(shard_files))
            print(f"  Shard {i}/{num_shards}: {len(shard_files)} files")

        total_files_all += len(valid_files)
        summary_categories[cat_name] = {
            "num_shards": num_shards,
            "total_files": len(valid_files),
            "shard_sizes": shard_sizes,
            "workers": num_workers,
        }
        print()

    summary = {
        "categories": summary_categories,
        "total_cases": None,
        "total_files_scanned": total_files_all,
    }
    summary_file = output_dir / "cases_collection_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to {summary_file}")

    print()
    print("=" * 80)
    print("Sharding Complete")
    print("=" * 80)
    for cat_name, cat_summary in summary_categories.items():
        sizes = cat_summary['shard_sizes']
        print(f"  {cat_name}: {cat_summary['total_files']} files -> {cat_summary['num_shards']} shards "
              f"(sizes: {sizes})")
    print(f"  Total: {total_files_all} files")


def parse_args():
    parser = argparse.ArgumentParser(description="Shard test files by business category")
    parser.add_argument("--test-dir", required=True, help="PyTorch test directory")
    parser.add_argument("--categories-config", required=True, help="Path to nightly_v2_test_whitelist.yml")
    parser.add_argument("--output-dir", required=True, help="Output directory for shard JSONs")
    return parser.parse_args()


if __name__ == "__main__":
    main()
