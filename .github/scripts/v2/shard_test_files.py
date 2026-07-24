#!/usr/bin/env python3
"""
Shard test files by business category using greedy LPT bin-packing.

Replaces collect_all_cases.py — no pytest --collect-only needed.
Reads category config YAML, validates file existence, then uses a
greedy Longest-Processing-Time (LPT) bin-packing algorithm (inspired
by upstream pytorch's tools/testing/test_selections.py::calculate_shards)
to distribute files across shards so that no shard exceeds the
configured max duration (default: 1 hour).

Falls back to round-robin when no test-times data is available.

Usage:
    python shard_test_files.py \
        --test-dir /path/to/pytorch/test \
        --categories-config .github/config/test_categories.yml \
        --test-times .github/config/test-times.json \
        --max-shard-duration 3600 \
        --output-dir /path/to/output
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None


def load_categories_config(config_path: str) -> Dict:
    """Load test_categories.yml and return the categories dict."""
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


def load_test_times(test_times_path: Optional[str]) -> Dict[str, Dict[str, float]]:
    """Load test-times.json.

    Expected format (similar to upstream pytorch's test-times.json but
    keyed by category instead of job_name/test_config):

        {
            "core":     { "test/nn/test_convolution.py": 267.7, ... },
            "tensor":   { "test/test_foreach.py": 479.1, ... },
            "distributed": { "test/distributed/test_c10d_gloo.py": 2249.7, ... },
            "graph":    { "test/test_meta.py": 3070.6, ... },
            "math":     { "test/test_linalg.py": 105.7 },
            "default":  { "test/any_file.py": 42.0, ... }
        }

    Returns empty dict if the file is missing or unreadable.
    """
    if not test_times_path:
        return {}
    p = Path(test_times_path).resolve()
    if not p.exists():
        print(f"  WARNING: test-times file not found: {p}", file=sys.stderr)
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            print(f"  WARNING: test-times file is not a JSON object: {p}", file=sys.stderr)
            return {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: Failed to load test-times file {p}: {e}", file=sys.stderr)
        return {}


def split_round_robin(files: List[str], num_shards: int) -> List[List[str]]:
    """Round-robin distribute files across shards for balanced load."""
    if num_shards <= 0:
        num_shards = 1
    shards = [[] for _ in range(num_shards)]
    for i, f in enumerate(sorted(files)):
        shards[i % num_shards].append(f)
    return shards


def split_greedy(
    files: List[str],
    durations: Dict[str, float],
    max_shard_duration: float,
    min_shards: int,
) -> Tuple[List[List[Dict]], int, List[float]]:
    """Greedy LPT (Longest Processing Time) bin-packing with sub-sharding.

    Inspired by upstream pytorch's ``get_with_pytest_shard`` and
    ``calculate_shards`` in tools/testing/test_selections.py.

    Files whose duration exceeds *max_shard_duration* are split into
    multiple sub-shard entries (each carrying ``shard_id`` /
    ``num_shards``).  The pytest plugin ``pytest_shard_custom.py`` then
    uses ``sha256(nodeid) % num_shards == shard_id - 1`` to partition
    cases within that file — so different sub-shards of the same file
    can be assigned to different CI machines safely.

    Algorithm:
      1. Expand each known-duration file into one or more sub-shard
         entries (like upstream ``get_with_pytest_shard``).  A file
         with ``duration > max_shard_duration`` is split into
         ``ceil(duration / max_shard_duration)`` entries.
      2. Sort entries by estimated duration descending (LPT).
      3. Calculate ``num_shards = max(min_shards, ceil(total / max))``.
      4. Greedily assign each entry to the shard with the smallest
         accumulated time.
      5. Distribute unknown-duration files round-robin.
      6. If any shard still exceeds *max_shard_duration* and has >1
         entry, increment ``num_shards`` and retry.

    Returns ``(shards, num_shards, shard_times)`` where each shard is a
    list of dicts: ``{"file", "shard_id", "num_shards", "estimated_duration"}``.
    """
    known_entries: List[Dict] = []
    unknown_files: List[str] = []

    for f in files:
        duration = durations.get(f, 0)
        if duration > 0:
            if duration > max_shard_duration:
                num_sub = math.ceil(duration / max_shard_duration)
                for i in range(num_sub):
                    known_entries.append({
                        "file": f,
                        "shard_id": i + 1,
                        "num_shards": num_sub,
                        "estimated_duration": duration / num_sub,
                    })
            else:
                known_entries.append({
                    "file": f,
                    "shard_id": 1,
                    "num_shards": 1,
                    "estimated_duration": duration,
                })
        else:
            unknown_files.append(f)

    known_entries.sort(key=lambda e: e["estimated_duration"], reverse=True)
    total_known = sum(e["estimated_duration"] for e in known_entries)

    if total_known > 0:
        num_shards = max(min_shards, math.ceil(total_known / max_shard_duration))
    else:
        num_shards = min_shards

    max_possible = max(min_shards, len(known_entries) + len(unknown_files))
    if num_shards > max_possible:
        num_shards = max_possible

    while True:
        shards: List[List[Dict]] = [[] for _ in range(num_shards)]
        shard_times = [0.0] * num_shards

        for entry in known_entries:
            min_idx = min(range(num_shards), key=lambda i: shard_times[i])
            shards[min_idx].append(entry)
            shard_times[min_idx] += entry["estimated_duration"]

        if not shard_times:
            break

        max_time = max(shard_times)
        if max_time <= max_shard_duration:
            break

        max_idx = shard_times.index(max_time)
        if len(shards[max_idx]) <= 1:
            break

        if num_shards >= max_possible:
            break
        num_shards += 1

    rr_idx = 0
    for f in unknown_files:
        shards[rr_idx % num_shards].append({
            "file": f,
            "shard_id": 1,
            "num_shards": 1,
            "estimated_duration": 0.0,
        })
        rr_idx += 1

    return shards, num_shards, shard_times


def save_shard_json(
    output_dir: Path,
    category: str,
    shard_num: int,
    num_shards: int,
    files: List,
    estimated_duration: Optional[float] = None,
) -> Path:
    """Save {category}_files_shard_{n}.json.

    *files* items can be:
      - plain string  → whole file (shard_id=1, num_shards=1)
      - dict          → sub-shard entry with file/shard_id/num_shards
    """
    serialized = []
    for entry in files:
        if isinstance(entry, dict):
            if entry.get("shard_id", 1) == 1 and entry.get("num_shards", 1) == 1:
                serialized.append(entry["file"])
            else:
                serialized.append({
                    "file": entry["file"],
                    "shard_id": entry["shard_id"],
                    "num_shards": entry["num_shards"],
                })
        else:
            serialized.append(entry)

    data = {
        "shard": shard_num,
        "num_shards": num_shards,
        "test_type": category,
        "total_files": len(serialized),
        "files": serialized,
    }
    if estimated_duration is not None:
        data["estimated_duration"] = round(estimated_duration, 1)
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

    test_times = load_test_times(args.test_times)
    max_shard_duration = args.max_shard_duration

    use_greedy = bool(test_times)
    if use_greedy:
        print(f"  Greedy LPT sharding enabled (max {max_shard_duration}s per shard)")
    else:
        print("  No test-times data — falling back to round-robin sharding")

    print("=" * 80)
    print("Sharding test files by business category")
    print("=" * 80)
    print(f"Test directory: {test_dir}")
    print(f"Categories: {len(categories)}")
    print()

    summary_categories = {}
    total_files_all = 0

    for cat_name, cat_config in categories.items():
        config_shards = cat_config.get("shards", 1)
        execution = cat_config.get("execution", "concurrent")
        raw_files = cat_config.get("files", [])

        print(f"--- Category: {cat_name} ---")
        print(f"  Configured files: {len(raw_files)}")
        print(f"  Configured shards: {config_shards}")
        print(f"  Execution: {execution}")

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
                "shard_durations": [],
                "total_duration": 0.0,
            }
            continue

        cat_durations: Dict[str, float] = {}
        if use_greedy:
            cat_durations = test_times.get(cat_name, {})
            if not cat_durations:
                cat_durations = test_times.get("default", {})
            found = sum(1 for f in valid_files if f in cat_durations)
            total_dur = sum(cat_durations.get(f, 0) for f in valid_files)
            print(f"  Duration data: {found}/{len(valid_files)} files, "
                  f"total {total_dur:.1f}s ({total_dur / 60:.1f}min)")

        if use_greedy and cat_durations:
            shards, num_shards, shard_times = split_greedy(
                valid_files,
                {f: cat_durations.get(f, 0) for f in valid_files},
                max_shard_duration,
                config_shards,
            )
            shard_durations = shard_times
        else:
            shards = split_round_robin(valid_files, config_shards)
            num_shards = config_shards
            shard_durations = [0.0] * num_shards

        sub_sharded = set()
        for shard_entries in shards:
            for entry in shard_entries:
                if isinstance(entry, dict) and entry.get("num_shards", 1) > 1:
                    sub_sharded.add(entry["file"])
        if sub_sharded:
            print(f"  Sub-sharded files ({len(sub_sharded)}):")
            for sf in sorted(sub_sharded):
                dur = cat_durations.get(sf, 0) if cat_durations else 0
                num_sub = math.ceil(dur / max_shard_duration) if dur > 0 else 1
                print(f"    {sf}: {dur:.0f}s -> {num_sub} sub-shards")

        shard_sizes = []
        for i, shard_files in enumerate(shards, 1):
            dur = shard_durations[i - 1] if i - 1 < len(shard_durations) else 0.0
            save_shard_json(output_dir, cat_name, i, num_shards, shard_files, dur)
            shard_sizes.append(len(shard_files))
            dur_str = f", ~{dur:.0f}s ({dur / 60:.1f}min)" if dur > 0 else ""
            print(f"  Shard {i}/{num_shards}: {len(shard_files)} entries{dur_str}")

        total_dur = sum(shard_durations)
        total_files_all += len(valid_files)
        summary_categories[cat_name] = {
            "num_shards": num_shards,
            "total_files": len(valid_files),
            "shard_sizes": shard_sizes,
            "shard_durations": [round(d, 1) for d in shard_durations],
            "total_duration": round(total_dur, 1),
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
        durs = cat_summary.get('shard_durations', [])
        total_dur = cat_summary.get('total_duration', 0)
        dur_info = f", ~{total_dur:.0f}s total" if total_dur > 0 else ""
        print(f"  {cat_name}: {cat_summary['total_files']} files -> {cat_summary['num_shards']} shards "
              f"(sizes: {sizes}){dur_info}")
    print(f"  Total: {total_files_all} files")


def parse_args():
    parser = argparse.ArgumentParser(description="Shard test files by business category")
    parser.add_argument("--test-dir", required=True, help="PyTorch test directory")
    parser.add_argument("--categories-config", required=True, help="Path to test_categories.yml")
    parser.add_argument("--test-times", default=None,
                        help="Path to test-times.json for greedy duration-based sharding")
    parser.add_argument("--max-shard-duration", type=float, default=3600,
                        help="Max estimated duration per shard in seconds (default: 3600 = 1h)")
    parser.add_argument("--output-dir", required=True, help="Output directory for shard JSONs")
    return parser.parse_args()


if __name__ == "__main__":
    main()
