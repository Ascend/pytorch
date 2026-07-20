#!/usr/bin/env python3
"""
Generate a consolidated markdown/json report for the NPU full test workflow.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import aggregation function from parse_test_results.py
import parse_test_results


def parse_args():
    parser = argparse.ArgumentParser(description="Generate consolidated NPU full test report")
    parser.add_argument("--reports-root", required=True, help="Root directory containing shard report files")
    parser.add_argument("--output-markdown", required=True, help="Path to write markdown report")
    parser.add_argument("--output-json", required=True, help="Path to write JSON report")
    parser.add_argument("--pytorch-version", required=True, help="PyTorch version string")
    parser.add_argument("--torch-npu-whl", required=True, help="torch_npu wheel URL")
    parser.add_argument("--patch-count", default="N/A", help="Applied patch count")
    parser.add_argument("--shard-matrix-json", required=True, help="JSON array of requested shard ids")
    parser.add_argument("--docker-image", default="N/A", help="Docker image used for test execution")
    parser.add_argument("--runner", default="N/A", help="Runner machine type")
    parser.add_argument("--special-reports-root", help="Root directory containing special test report files")
    parser.add_argument("--expected-special-tests-json", default="[]", help="JSON array of expected special test names")
    parser.add_argument("--cases-summary", help="Path to cases_collection_summary.json for file discovery stats")
    return parser.parse_args()


def load_json_file(path: Path) -> Dict:
    """Load JSON file with error handling for malformed/truncated files."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {path}: {e}")
        # Read file content to diagnose truncation
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"  File size: {len(content)} bytes")
            # Show context around error position
            error_pos = e.pos if hasattr(e, 'pos') else 0
            start = max(0, error_pos - 100)
            end = min(len(content), error_pos + 100)
            print(f"  Context around error (pos {error_pos}): ...{content[start:end]}...")
        except Exception:
            pass
        return {}
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return {}


def parse_requested_shards(raw: str) -> List[Tuple[str, int]]:
    """
    Parse shard identifiers from JSON array.

    Supports formats:
    - Integers: [1, 2, 3] -> [("regular", 1), ("regular", 2), ("regular", 3)]
    - Type-prefixed: ["dist-1", "reg-2"] -> [("distributed", 1), ("regular", 2)]

    Returns list of (shard_type, shard_number) tuples.
    """
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(value, list):
        return []

    result = []
    for item in value:
        try:
            if isinstance(item, str):
                # Parse type-prefixed format: "dist-1", "reg-2"
                if "-" in item:
                    type_prefix, num_str = item.split("-", 1)
                    if type_prefix == "dist":
                        shard_type = "distributed"
                    elif type_prefix == "reg":
                        shard_type = "regular"
                    else:
                        # Unknown prefix, skip
                        continue
                    shard_num = int(num_str)
                    result.append((shard_type, shard_num))
                else:
                    # String without prefix, try to parse as int
                    shard_num = int(item)
                    result.append(("regular", shard_num))
            elif isinstance(item, int):
                # Plain integer, assume "regular" type
                result.append(("regular", item))
        except (TypeError, ValueError):
            continue
    # Sort by type then number
    return sorted(set(result), key=lambda x: (x[0], x[1]))


def parse_expected_special_tests(raw: str) -> List[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(value, list):
        return []

    result = []
    for item in value:
        if isinstance(item, str) and item:
            result.append(item)
    return sorted(set(result))


def load_text_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_int_value(payload: Dict, *keys: str) -> int:
    for key in keys:
        if key not in payload:
            continue
        try:
            return int(payload.get(key, 0))
        except (TypeError, ValueError):
            continue
    return 0


def discover_shard_files(
    reports_root: Path,
) -> Tuple[
    Dict[Tuple[str, int], Path],  # stats_files
    Dict[Tuple[str, int], Path],  # info_files
    Dict[Tuple[str, int], Path],  # cases_files
]:
    """
    Discover all shard report files in the reports directory.

    Returns dicts keyed by (shard_type, shard_number) tuples.

    File name format: shard_{type}-{number}_{suffix}
    Examples:
    - shard_dist-1_stats.json
    - shard_reg-1_info.json
    - shard_dist-1_cases.json  (case-level results)
    """
    stats_files = {}
    info_files = {}
    cases_files = {}

    def parse_shard_filename(path: Path, suffix_pattern: str) -> Tuple[str, int]:
        """
        Parse shard type and number from filename.

        Filename format: shard_{type}-{number}_{suffix}
        e.g., shard_dist-1_stats.json -> ("distributed", 1)
        """
        stem = path.stem  # filename without extension
        # Match pattern: shard_{type}-{number}_{suffix}
        match = re.match(r"shard_(dist|reg)-(\d+)_" + suffix_pattern, stem)
        if match:
            type_prefix = match.group(1)
            shard_num = int(match.group(2))
            if type_prefix == "dist":
                return ("distributed", shard_num)
            elif type_prefix == "reg":
                return ("regular", shard_num)
        return None

    for path in reports_root.rglob("shard_*_stats.json"):
        key = parse_shard_filename(path, "stats")
        if key:
            stats_files[key] = path

    for path in reports_root.rglob("shard_*_info.json"):
        key = parse_shard_filename(path, "info")
        if key:
            info_files[key] = path

    # Discover case-level results files
    for path in reports_root.rglob("shard_*_cases.json"):
        key = parse_shard_filename(path, "cases")
        if key:
            cases_files[key] = path

    return stats_files, info_files, cases_files


def build_file_to_shards_map(cases_shards_dir: Path) -> Dict[str, List[str]]:
    """
    Build a mapping from test file path to shard IDs.

    Scans all shard JSON files in cases_shards_dir and extracts file->shard mapping.

    Args:
        cases_shards_dir: Directory containing shard JSON files like
                          distributed_cases_shard_1.json, regular_cases_shard_2.json

    Returns:
        Dict mapping file path (e.g., "test/test_ops.py") to list of shard IDs
        (e.g., ["dist-1", "reg-2", "reg-3"])
    """
    file_to_shards = {}

    if not cases_shards_dir or not cases_shards_dir.exists():
        return file_to_shards

    # Pattern: {test_type}_cases_shard_{num}.json
    for shard_file in cases_shards_dir.glob("*_cases_shard_*.json"):
        try:
            data = load_json_file(shard_file)
            test_type = data.get("test_type", "regular")
            shard_num = data.get("shard", 0)

            # Build shard ID: "dist-1" or "reg-2"
            shard_prefix = "dist" if test_type == "distributed" else "reg"
            shard_id = f"{shard_prefix}-{shard_num}"

            # Extract file paths from cases
            cases = data.get("cases", [])
            for case in cases:
                file_path = case.get("file", "")
                if file_path:
                    # Normalize file path (remove leading "test/" if present for consistency)
                    normalized_file = file_path
                    if normalized_file.startswith("test/"):
                        normalized_file = normalized_file[5:]

                    if normalized_file not in file_to_shards:
                        file_to_shards[normalized_file] = []
                    if shard_id not in file_to_shards[normalized_file]:
                        file_to_shards[normalized_file].append(shard_id)
        except Exception as e:
            print(f"Warning: Failed to parse shard file {shard_file}: {e}")
            continue

    # Sort shard IDs for each file
    for file_path in file_to_shards:
        # Sort by type (dist first) then number
        file_to_shards[file_path].sort(key=lambda x: (0 if x.startswith("dist") else 1, int(x.split("-")[1])))

    return file_to_shards


def get_shard_status(stats: Dict, present: bool) -> str:
    if not present:
        return "MISSING"
    if stats.get("timed_out"):
        return "TIMEOUT"
    if stats.get("incomplete"):
        return "INCOMPLETE"
    if stats.get("errors", 0) > 0:
        return "ERROR"
    if stats.get("failed", 0) > 0:
        return "FAILED"
    if stats.get("total", 0) == 0:
        return "NO TESTS"
    return "PASSED"


def get_overall_status(status_counts: Counter) -> str:
    if status_counts["MISSING"] > 0:
        return "FAILED"
    if any(status_counts[key] > 0 for key in ("TIMEOUT", "INCOMPLETE", "ERROR", "FAILED")):
        return "FAILED"
    if status_counts["PASSED"] > 0:
        return "PASSED"
    return "NO TESTS"


def format_duration(seconds: float) -> str:
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    if minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    return f"{secs:.1f}s"


def sanitize_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def render_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def discover_special_test_files(reports_root: Path | None) -> Dict[str, Path]:
    if reports_root is None or not reports_root.exists():
        return {}

    special_files = {}
    for path in reports_root.rglob("special_test_*.json"):
        try:
            payload = load_json_file(path)
        except Exception:
            continue
        name = payload.get("name")
        if isinstance(name, str) and name:
            special_files[name] = path
    return special_files


def main():
    args = parse_args()
    reports_root = Path(args.reports_root)
    output_markdown = Path(args.output_markdown)
    output_json = Path(args.output_json)
    requested_shards = parse_requested_shards(args.shard_matrix_json)
    expected_special_tests = parse_expected_special_tests(args.expected_special_tests_json)
    special_reports_root = Path(args.special_reports_root) if args.special_reports_root else None

    # Load cases collection summary for file discovery stats
    cases_summary_data = None
    file_discovery_stats = {
        "total_files_scanned": 0,
        "distributed_files": 0,
        "regular_files": 0,
    }
    if args.cases_summary:
        cases_summary_path = Path(args.cases_summary)
        if cases_summary_path.exists():
            cases_summary_data = load_json_file(cases_summary_path)
            # Extract file discovery stats (正交: total = distributed + regular)
            if cases_summary_data:
                file_discovery_stats["total_files_scanned"] = cases_summary_data.get("total_files_scanned", 0)
                file_discovery_stats["distributed_files"] = cases_summary_data.get("distributed_files", 0)
                file_discovery_stats["regular_files"] = cases_summary_data.get("regular_files", 0)

    stats_files, info_files, cases_files = discover_shard_files(reports_root)
    special_test_files = discover_special_test_files(special_reports_root)
    shard_ids = requested_shards or sorted(set(stats_files) | set(info_files) | set(cases_files))

    # Build file to shards mapping from cases-shards directory
    cases_shards_dir = Path(args.cases_summary).parent if args.cases_summary else None
    file_to_shards_map = build_file_to_shards_map(cases_shards_dir)

    status_counts = Counter()
    totals = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "timeout": 0,
        "duration": 0.0,
    }
    shard_rows = []
    selection_modes = set()
    cases_results = {}  # Store case-level results for each shard

    for shard_type, shard_num in shard_ids:
        shard_key = (shard_type, shard_num)
        stats_path = stats_files.get(shard_key)
        info_path = info_files.get(shard_key)
        cases_path = cases_files.get(shard_key)
        stats = load_json_file(stats_path) if stats_path else {}
        info = load_json_file(info_path) if info_path else {}

        # Load case-level results if available
        cases_data = load_json_file(cases_path) if cases_path else {}
        if cases_data:
            cases_results[shard_key] = cases_data
            # Override stats with case-level data
            stats["total"] = cases_data.get("total_cases", 0)
            stats["passed"] = cases_data.get("passed", 0)
            stats["failed"] = cases_data.get("failed", 0)
            stats["errors"] = cases_data.get("errors", 0)
            stats["skipped"] = cases_data.get("skipped", 0)
            stats["timeout"] = cases_data.get("timeout", 0)
            stats["duration"] = cases_data.get("duration", 0.0)
            # Update totals (正交累加: total = passed + failed + errors + skipped + timeout)
            totals["total"] += cases_data.get("total_cases", 0)
            totals["passed"] += cases_data.get("passed", 0)
            totals["failed"] += cases_data.get("failed", 0)
            totals["errors"] += cases_data.get("errors", 0)
            totals["skipped"] += cases_data.get("skipped", 0)
            totals["timeout"] += cases_data.get("timeout", 0)
            totals["duration"] += cases_data.get("duration", 0.0)

        present = bool(stats_path or cases_path)

        if info.get("selection_mode"):
            selection_modes.add(str(info.get("selection_mode")))

        status = get_shard_status(stats, present)
        status_counts[status] += 1

        # Convert shard_type to display prefix ("distributed" -> "dist", "regular" -> "reg")
        shard_prefix = "dist" if shard_type == "distributed" else "reg"
        shard_rows.append(
            {
                "shard": f"{shard_prefix}-{shard_num}",  # "dist-1" or "reg-1"
                "shard_type": shard_type,
                "shard_num": shard_num,
                "status": status,
                "total": int(stats.get("total", 0)),
                "passed": int(stats.get("passed", 0)),
                "failed": int(stats.get("failed", 0)),
                "skipped": int(stats.get("skipped", 0)),
                "errors": int(stats.get("errors", 0)),
                "timeout": int(stats.get("timeout", 0)),
                "duration": float(stats.get("duration", 0.0)),
            }
        )

    overall_status = get_overall_status(status_counts)
    whl_name = Path(args.torch_npu_whl).name
    received_reports = len(stats_files)
    expected_reports = len(shard_ids)
    selection_mode_display = ", ".join(sorted(selection_modes)) if selection_modes else "-"

    # Show all shards in the detail table
    sorted_shards = sorted(shard_rows, key=lambda row: (row["shard_type"], row["shard_num"]))
    special_test_names = expected_special_tests or sorted(special_test_files)
    special_test_rows = []
    special_status_counts = Counter()

    for test_name in special_test_names:
        payload = load_json_file(special_test_files[test_name]) if test_name in special_test_files else {}
        status = str(payload.get("status", "MISSING"))
        special_status_counts[status] += 1
        special_test_rows.append(
            {
                "name": test_name,
                "group": str(payload.get("group", "-")),
                "status": status,
                "duration": float(payload.get("duration", 0.0)),
                "returncode": payload.get("returncode", "-"),
                "note": str(payload.get("note", "") or "-"),
            }
        )

    if any(row["status"] != "PASSED" for row in special_test_rows):
        overall_status = "FAILED"

    include_special_tests = bool(special_test_names or special_test_rows)

    # Build Selection row content based on available data
    if cases_summary_data:
        # Use file discovery stats from cases_collection_summary.json
        total_scanned = file_discovery_stats["total_files_scanned"]
        dist_files = file_discovery_stats["distributed_files"]
        reg_files = file_discovery_stats["regular_files"]
        selection_content = (
            f"扫描发现 {total_scanned} 个测试文件 "
            f"(distributed: {dist_files}, regular: {reg_files})"
        )
    else:
        # Fallback to original selection mode display
        selection_content = selection_mode_display

    # Extract planned cases count from cases_collection_summary.json
    planned_total_cases = 0
    planned_dist_cases = 0
    planned_reg_cases = 0
    if cases_summary_data:
        planned_total_cases = cases_summary_data.get("total_cases", 0)
        planned_dist_cases = cases_summary_data.get("distributed", {}).get("cases_summary", {}).get("total_cases", 0)
        planned_reg_cases = cases_summary_data.get("regular", {}).get("cases_summary", {}).get("total_cases", 0)

    overview_rows = [
        ["Overall result", overall_status],
        ["PyTorch", f"`v{args.pytorch_version}`"],
        ["torch_npu", f"`{whl_name}`"],
        ["Patches applied", str(args.patch_count)],
        ["Docker image", f"`{args.docker_image}`"],
        ["Runner", f"`{args.runner}`"],
        ["Shards", f"{received_reports} / {expected_reports} reported"],
        ["Selection", selection_content],
        [
            "实际执行用例",
            (
                f"{totals['total']} total; {totals['passed']} passed; {totals['failed']} failed; "
                f"{totals['errors']} errors; {totals['skipped']} skipped; "
                f"{totals['timeout']} timeout"
            ),
        ],
    ]
    # Add planned cases count row if available
    if planned_total_cases > 0:
        overview_rows.append([
            "规划用例总数",
            f"{planned_total_cases} (distributed: {planned_dist_cases}, regular: {planned_reg_cases})",
        ])
    overview_rows.append(["Duration", format_duration(totals["duration"])])
    if include_special_tests:
        overview_rows.append(["Special tests expected", str(len(special_test_names))])

    markdown_lines = [
        "# PyTorch NPU Full Test Summary",
        "",
        "## Overview",
    ]
    markdown_lines.extend(
        render_table(
            ["Item", "Value"],
            overview_rows,
        )
    )

    # Add case-level statistics table if available
    if cases_results:
        markdown_lines.extend(["", "## 用例级执行统计"])
        markdown_lines.extend(
            render_table(
                ["Shard", "总用例", "通过", "失败", "错误", "跳过", "超时", "Duration"],
                [
                    [
                        f"{row['shard']}",
                        str(row["total"]),
                        str(row["passed"]),
                        str(row["failed"]),
                        str(row["errors"]),
                        str(row.get("skipped", 0)),
                        str(row.get("timeout", 0)),
                        format_duration(row["duration"]),
                    ]
                    for row in sorted_shards
                    if (row["shard_type"], row["shard_num"]) in cases_results
                ],
            )
        )

        # Add file-level statistics table
        file_stats = parse_test_results.aggregate_all_cases_by_file(cases_results)

        if file_stats:
            # Sort files by total cases descending
            sorted_files = sorted(
                file_stats.values(),
                key=lambda x: (-x["total"], x["file"])
            )

            markdown_lines.extend(["", "## 测试文件结果汇总"])

            file_rows = []
            for fs in sorted_files:  # Show all files
                failed_total = fs["failed"] + fs["errors"] + fs["timeout"]
                fail_rate = f"{(failed_total / fs['total'] * 100):.1f}%" if fs["total"] > 0 else "0%"
                # Get shard info for this file
                file_path = fs["file"]
                # Normalize file path for lookup (remove leading "test/")
                lookup_path = file_path
                if lookup_path.startswith("test/"):
                    lookup_path = lookup_path[5:]
                shards_for_file = file_to_shards_map.get(lookup_path, [])
                shard_info = ", ".join(shards_for_file) if shards_for_file else "-"
                file_rows.append([
                    sanitize_markdown_cell(fs["file"]),
                    shard_info,
                    str(fs["total"]),
                    str(fs["passed"]),
                    str(fs["failed"]),
                    str(fs["errors"]),
                    str(fs["skipped"]),
                    str(fs["timeout"]),
                    fail_rate,
                ])

            markdown_lines.extend(
                render_table(
                    ["测试文件", "分片", "总用例", "通过", "失败", "错误", "跳过", "超时", "失败率"],
                    file_rows,
                )
            )

    if include_special_tests:
        markdown_lines.extend(["", "## Special Test Results"])
        markdown_lines.extend(
            render_table(
                ["Test", "Group", "Status", "Duration", "Return Code", "Note"],
                [
                    [
                        row["name"],
                        row["group"],
                        row["status"],
                        format_duration(row["duration"]),
                        str(row["returncode"]),
                        sanitize_markdown_cell(row["note"]),
                    ]
                    for row in special_test_rows
                ] or [["-", "-", "-", "0.0s", "-", "-"]],
            )
        )

    report_json = {
        "overall_status": overall_status,
        "requested_shards": shard_ids,
        "reports_collected": received_reports,
        "patch_count": args.patch_count,
        "pytorch_version": args.pytorch_version,
        "torch_npu_whl": whl_name,
        "docker_image": args.docker_image,
        "runner": args.runner,
        "status_counts": dict(status_counts),
        "totals": totals,
        "file_discovery_stats": file_discovery_stats,
        "planned_cases": {
            "total": planned_total_cases,
            "distributed": planned_dist_cases,
            "regular": planned_reg_cases,
        },
        "shards": shard_rows,
    }

    # Add full cases summary if available
    if cases_summary_data:
        report_json["cases_collection_summary"] = cases_summary_data

    # Add case-level results if available
    if cases_results:
        report_json["cases_results"] = {
            "shards": {
                f"{shard_type}-{shard_num}": data
                for (shard_type, shard_num), data in cases_results.items()
            },
        }

        # Add file-level aggregation
        file_stats = parse_test_results.aggregate_all_cases_by_file(cases_results)
        # Add shard info to file stats
        file_stats_with_shards = {}
        for file_path, stats in file_stats.items():
            # Normalize file path for lookup
            lookup_path = file_path
            if lookup_path.startswith("test/"):
                lookup_path = lookup_path[5:]
            shards_for_file = file_to_shards_map.get(lookup_path, [])
            stats["shards"] = shards_for_file
            file_stats_with_shards[file_path] = stats
        report_json["file_level_stats"] = dict(sorted(
            file_stats_with_shards.items(),
            key=lambda x: (-x[1]["total"], x[0])
        ))

        # Add list of files with failures
        failed_files = parse_test_results.get_files_with_failures(file_stats)
        report_json["files_with_failures"] = failed_files

    if include_special_tests:
        report_json["special_tests"] = {
            "expected": special_test_names,
            "status_counts": dict(special_status_counts),
            "results": special_test_rows,
        }

    output_markdown.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    output_json.write_text(json.dumps(report_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Generated markdown report: {output_markdown}")
    print(f"Generated json report: {output_json}")


if __name__ == "__main__":
    main()
