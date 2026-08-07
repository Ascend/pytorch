#!/usr/bin/env python3
"""
Generate test-times.json from shard case result JSONs.

Aggregates per-file execution timing data from all shard results
across every category, producing a single JSON mapping:
    category_name -> {file_path: total_seconds}

Usage:
    python generate_test_times.py \
        --reports-root all-test-reports \
        --output test-times.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    args = parse_args()
    reports_root = Path(args.reports_root)

    if not reports_root.is_dir():
        print(f"Reports root not found: {reports_root}", file=sys.stderr)
        sys.exit(0)

    all_times = {"default": {}}

    # Walk all JSON/JSONL files and collect per-file timing
    for fpath in reports_root.rglob("*.json"):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # Handle shard cases JSON: has "files" or "cases" with per-file timing
        files_data = data.get("files", [])
        if not files_data and "cases" in data:
            # Try cases format
            files_data = []
            for case in data.get("cases", []):
                fn = case.get("file", "")
                if fn:
                    files_data.append(case)

        if not files_data:
            continue

        category = data.get("test_type", data.get("category", "default"))
        if category not in all_times:
            all_times[category] = {}

        for entry in files_data:
            fn = entry.get("file", entry.get("test_file", ""))
            elapsed = entry.get("elapsed", entry.get("time", entry.get("duration", 0)))
            if fn and elapsed:
                # Keep cumulative time (sum across shards)
                all_times[category][fn] = (
                    all_times[category].get(fn, 0) + float(elapsed)
                )

    # Also handle JSONL format (shard_*-*_cases.jsonl)
    for fpath in reports_root.rglob("*.jsonl"):
        try:
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "_meta" in obj:
                        continue
                    fn = obj.get("file", "")
                    elapsed = obj.get("elapsed", obj.get("time", 0))
                    if fn and elapsed:
                        cat = obj.get("category", "default")
                        if cat not in all_times:
                            all_times[cat] = {}
                        all_times[cat][fn] = (
                            all_times[cat].get(fn, 0) + float(elapsed)
                        )
        except (json.JSONDecodeError, OSError):
            continue

    # Write output
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(all_times, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Print summary
    print("=== test-times.json preview ===")
    for cat in sorted(k for k in all_times if k != "default"):
        files = all_times[cat]
        if files:
            total = sum(files.values())
            print(f"  {cat}: {len(files)} files, {total:.0f}s ({total / 60:.1f}min)")

    if not any(all_times.get(c) for c in all_times if c != "default"):
        print("  (no per-file timing data found)")
        all_times["default"] = {}

    print(f"Saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-file test timing data from shard reports"
    )
    parser.add_argument(
        "--reports-root",
        required=True,
        help="Root directory containing downloaded test-reports artifacts",
    )
    parser.add_argument(
        "--output",
        default="test-times.json",
        help="Output JSON file path (default: test-times.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
