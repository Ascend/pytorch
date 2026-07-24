#!/usr/bin/env python3
"""
Aggregate per-file wall-clock execution times from all shard artifacts
and generate an updated test-times.json for the next pipeline run.

Reads ``file_execution_times.json`` files from the ``all-test-reports/``
directory (downloaded from all test-reports-* artifacts via
``merge-multiple: true``).  Each file was written by ``run_npu_test_file.py``
and contains real wall-clock timing including process startup, NPU init,
collection, fixture setup, and all crash-retry attempts.

For sub-sharded files (num_shards > 1), the wall times from each
sub-shard are summed to reconstruct the full file duration.

Usage:
    python generate_test_times.py \
        --reports-root all-test-reports \
        --output test-times.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict


def load_file_execution_times(reports_root: Path) -> list:
    """Find and load all file_execution_times.json files."""
    results = []
    for p in sorted(reports_root.rglob("file_execution_times.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            results.append(data)
            shard = data.get("shard", "?")
            shard_type = data.get("shard_type", "?")
            num_files = len(data.get("file_times", []))
            print(f"  Loaded {p.name}: shard {shard_type}-{shard}, {num_files} files", flush=True)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Failed to load {p}: {e}", flush=True)
    return results


def aggregate_times(all_shard_data: list) -> Dict[str, Dict[str, float]]:
    """Aggregate per-file wall times into the test-times.json format.

    For sub-sharded files (num_shards > 1), sum wall_time across all
    sub-shards to reconstruct the full file duration.

    Returns dict keyed by shard_type, then by file path:
        { "core": { "test/foo.py": 123.4, ... }, ... }
    Also includes a "default" key with all files merged.
    """
    # file_path -> { shard_type -> total_wall_time }
    file_durations: Dict[str, Dict[str, float]] = {}

    for shard_data in all_shard_data:
        shard_type = shard_data.get("shard_type", "unknown")
        for entry in shard_data.get("file_times", []):
            file_path = entry["file"]
            wall_time = entry.get("wall_time", 0.0)
            sub_num_shards = entry.get("num_shards", 1)

            if file_path not in file_durations:
                file_durations[file_path] = {}

            if sub_num_shards > 1:
                file_durations[file_path][shard_type] = \
                    file_durations[file_path].get(shard_type, 0.0) + wall_time
            else:
                file_durations[file_path][shard_type] = wall_time

    # Build per-category and default dicts
    result: Dict[str, Dict[str, float]] = {}
    all_files: Dict[str, float] = {}

    for file_path, cat_times in file_durations.items():
        # A file may appear in multiple categories (e.g. moved between runs).
        # Use the max across categories as the canonical duration.
        max_duration = max(cat_times.values()) if cat_times else 0.0
        all_files[file_path] = round(max_duration, 1)

        for cat, dur in cat_times.items():
            if cat not in result:
                result[cat] = {}
            result[cat][file_path] = round(dur, 1)

    result["default"] = dict(sorted(all_files.items()))

    return result


def main():
    args = parse_args()

    reports_root = Path(args.reports_root).resolve()
    if not reports_root.is_dir():
        raise FileNotFoundError(f"Reports root not found: {reports_root}")

    print(f"Scanning {reports_root} for file_execution_times.json...", flush=True)
    all_shard_data = load_file_execution_times(reports_root)

    if not all_shard_data:
        print("WARNING: No file_execution_times.json files found. "
              "Output will be empty.", flush=True)

    print(f"\nAggregating timings from {len(all_shard_data)} shard reports...", flush=True)
    test_times = aggregate_times(all_shard_data)

    # Print summary
    print(f"\n{'=' * 80}")
    print("Test Times Summary")
    print(f"{'=' * 80}")
    for cat in sorted(k for k in test_times if k != "default"):
        files = test_times[cat]
        total = sum(files.values())
        print(f"  {cat}: {len(files)} files, total {total:.0f}s ({total / 60:.1f}min)")
        for f, d in sorted(files.items(), key=lambda x: -x[1])[:5]:
            print(f"    {d:>8.1f}s  {f}")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more")
    default = test_times.get("default", {})
    default_total = sum(default.values())
    print(f"\n  default: {len(default)} files, total {default_total:.0f}s "
          f"({default_total / 60:.1f}min)")

    # Write output
    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(test_times, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTest times written to {output_path} ({output_path.stat().st_size} bytes)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-file wall-clock times into test-times.json"
    )
    parser.add_argument("--reports-root", required=True,
                        help="Root directory containing file_execution_times.json files")
    parser.add_argument("--output", required=True,
                        help="Output path for test-times.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
