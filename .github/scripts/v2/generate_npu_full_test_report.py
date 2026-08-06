#!/usr/bin/env python3
"""
Generate a consolidated markdown/json report for the NPU full test workflow.

Output files:
- npu-full-test-summary.json: Lightweight summary with aggregated stats only
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# Status Constants
# ==============================================================================

STATUS_MISSING = "MISSING"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_ERROR = "ERROR"
STATUS_FAILED = "FAILED"
STATUS_PASSED = "PASSED"
STATUS_NO_TESTS = "NO TESTS"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate consolidated NPU full test report")
    parser.add_argument("--reports-root", required=True, help="Root directory containing shard report files")
    parser.add_argument("--output-markdown", required=True, help="Path to write markdown report")
    parser.add_argument("--output-jsonl", required=True, help="Path to write aggregated JSONL summary")
    parser.add_argument("--pytorch-version", required=True, help="PyTorch version string")
    parser.add_argument("--torch-npu-whl", required=True, help="torch_npu wheel URL")
    parser.add_argument("--patch-count", default="N/A", help="Applied patch count")
    parser.add_argument("--shard-matrix-json", required=True, help="JSON array of requested shard ids")
    parser.add_argument("--docker-image", default="N/A", help="Docker image used for test execution")
    parser.add_argument("--special-reports-root", help="Root directory containing special test report files")
    parser.add_argument("--expected-special-tests-json", default="[]", help="JSON array of expected special test names")
    return parser.parse_args()


def load_json_file(path: Path) -> Dict:
    """Load JSON file with error handling for malformed/truncated files."""
    try:
        content = path.read_text(encoding="utf-8")
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {path}: {e}")
        print(f"  File size: {len(content)} bytes")
        # Show context around error position
        error_pos = e.pos if hasattr(e, 'pos') else 0
        start = max(0, error_pos - 100)
        end = min(len(content), error_pos + 100)
        print(f"  Context around error (pos {error_pos}): ...{content[start:end]}...")
        return {}
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return {}


def parse_requested_shards(raw: str) -> List[Tuple[str, int]]:
    """
    Parse shard identifiers from JSON array.

    Supports formats:
    - Integers: [1, 2, 3] -> [("regular", 1), ("regular", 2), ("regular", 3)]
    - Type-prefixed: ["dist-1", "reg-2", "custom-1", "core-1", "tensor-1", "graph-1", "others-1"]

    Returns list of (shard_type, shard_number) tuples.
    """
    _PREFIX_TO_TYPE = {
        "dist": "distributed",
        "reg": "regular",
        "custom": "custom",
        "core": "core",
        "tensor": "tensor",
        "graph": "graph",
        "math": "math",
        "others": "others",
    }
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
                # Parse type-prefixed format: "dist-1", "core-2", etc.
                if "-" in item:
                    type_prefix, num_str = item.split("-", 1)
                    shard_type = _PREFIX_TO_TYPE.get(type_prefix)
                    if shard_type is None:
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


def discover_shard_files(
    reports_root: Path,
) -> Dict[Tuple[str, int], Path]:
    # returns cases_files
    """
    Discover all shard report files in the reports directory.

    Returns dicts keyed by (shard_type, shard_number) tuples.

    File name format: shard_{type}-{number}_{suffix}
    Examples:
    - shard_dist-1_stats.json
    - shard_reg-1_info.json
    - shard_dist-1_cases.json  (case-level results)
    """
    cases_files = {}

    def parse_shard_filename(path: Path, suffix_pattern: str) -> Optional[Tuple[str, int]]:
        """
        Parse shard type and number from filename.

        Filename format: shard_{type}-{number}_{suffix}
        e.g., shard_dist-1_stats.json -> ("distributed", 1)
              shard_reg-1_stats.json -> ("regular", 1)
              shard_core-1_stats.json -> ("core", 1)
              shard_tensor-1_stats.json -> ("tensor", 1)
              shard_graph-1_stats.json -> ("graph", 1)
              shard_math-1_stats.json -> ("math", 1)
        """
        _PREFIX_TO_TYPE = {
            "dist": "distributed",
            "reg": "regular",
            "custom": "custom",
            "core": "core",
            "tensor": "tensor",
            "graph": "graph",
            "math": "math",
        "others": "others",
        }
        stem = path.stem  # filename without extension
        # Match pattern: shard_{type}-{number}_{suffix}
        match = re.match(r"shard_(dist|reg|custom|core|tensor|graph|math|others)-(\d+)_" + suffix_pattern, stem)
        if match:
            type_prefix = match.group(1)
            shard_num = int(match.group(2))
            shard_type = _PREFIX_TO_TYPE.get(type_prefix)
            if shard_type:
                return (shard_type, shard_num)
        return None

    for path in reports_root.rglob("shard_*_cases.jsonl"):
        key = parse_shard_filename(path, "cases")
        if key:
            cases_files[key] = path

    return cases_files


def get_shard_status(stats: Dict, present: bool) -> str:
    if not present:
        return STATUS_MISSING
    if stats.get("timed_out"):
        return STATUS_TIMEOUT
    if stats.get("incomplete"):
        return STATUS_INCOMPLETE
    if stats.get("errors", 0) > 0:
        return STATUS_ERROR
    if stats.get("failed", 0) > 0:
        return STATUS_FAILED
    if stats.get("total", 0) == 0:
        return STATUS_NO_TESTS
    return STATUS_PASSED


def get_overall_status(status_counts: Counter) -> str:
    if status_counts[STATUS_MISSING] > 0:
        return STATUS_FAILED
    if any(status_counts[key] > 0 for key in (STATUS_TIMEOUT, STATUS_INCOMPLETE, STATUS_ERROR, STATUS_FAILED)):
        return STATUS_FAILED
    if status_counts[STATUS_PASSED] > 0:
        return STATUS_PASSED
    return STATUS_NO_TESTS


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
    output_jsonl = Path(args.output_jsonl)
    requested_shards = parse_requested_shards(args.shard_matrix_json)
    expected_special_tests = parse_expected_special_tests(args.expected_special_tests_json)
    special_reports_root = Path(args.special_reports_root) if args.special_reports_root else None

    cases_files = discover_shard_files(reports_root)
    special_test_files = discover_special_test_files(special_reports_root)
    shard_ids = requested_shards or sorted(set(cases_files))

    status_counts = Counter()
    totals = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "duration": 0.0,
    }
    shard_rows = []
    all_file_records = []
    execution_modes = set()
    runners = set()

    for shard_type, shard_num in shard_ids:
        shard_key = (shard_type, shard_num)
        cases_path = cases_files.get(shard_key)
        stats = {}

        # Read JSONL: first line = shard summary, lines 2+ = per-file records
        cases_data = {}
        if cases_path:
            try:
                with open(cases_path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        cases_data = json.loads(first_line)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        rec["shard_type"] = shard_type
                        rec["shard"] = shard_num
                        all_file_records.append(rec)
            except Exception:
                pass
        if cases_data:
            stats["total"] = cases_data.get("total_cases", 0)
            stats["passed"] = cases_data.get("passed", 0)
            stats["failed"] = cases_data.get("failed", 0)
            stats["errors"] = cases_data.get("errors", 0)
            stats["skipped"] = cases_data.get("skipped", 0)
            stats["duration"] = cases_data.get("duration", 0.0)
            totals["total"] += cases_data.get("total_cases", 0)
            totals["passed"] += cases_data.get("passed", 0)
            totals["failed"] += cases_data.get("failed", 0)
            totals["errors"] += cases_data.get("errors", 0)
            totals["skipped"] += cases_data.get("skipped", 0)
            totals["duration"] += cases_data.get("duration", 0.0)

        present = bool(cases_path)

        if cases_data.get("execution_mode"):
            execution_modes.add(str(cases_data["execution_mode"]))
        if cases_data.get("runner"):
            runners.add(str(cases_data["runner"]))

        status = get_shard_status(stats, present)
        status_counts[status] += 1

        # Convert shard_type to display prefix
        _TYPE_TO_PREFIX = {
            "distributed": "dist",
            "regular": "reg",
            "custom": "custom",
            "core": "core",
            "tensor": "tensor",
            "graph": "graph",
            "math": "math",
        "others": "others",
        }
        shard_prefix = _TYPE_TO_PREFIX.get(shard_type, "reg")
        shard_rows.append(
            {
                "shard": f"{shard_prefix}-{shard_num}",  # "dist-1", "reg-1", or "custom-1"
                "shard_type": shard_type,
                "shard_num": shard_num,
                "status": status,
                "total": int(stats.get("total", 0)),
                "passed": int(stats.get("passed", 0)),
                "failed": int(stats.get("failed", 0)),
                "skipped": int(stats.get("skipped", 0)),
                "errors": int(stats.get("errors", 0)),
                "duration": float(stats.get("duration", 0.0)),
            }
        )

    overall_status = get_overall_status(status_counts)
    whl_name = Path(args.torch_npu_whl).name
    received_reports = len(cases_files)
    expected_reports = len(shard_ids)
    selection_mode_display = ", ".join(sorted(execution_modes)) if execution_modes else "-"
    runner_display = ", ".join(sorted(runners)) if runners else "-"

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

    if any(row["status"] != STATUS_PASSED for row in special_test_rows):
        overall_status = STATUS_FAILED

    include_special_tests = bool(special_test_names or special_test_rows)
    selection_content = selection_mode_display

    overview_rows = [
        ["Overall result", overall_status],
        ["PyTorch", f"`v{args.pytorch_version}`"],
        ["torch_npu", f"`{whl_name}`"],
        ["Patches applied", str(args.patch_count)],
        ["Docker image", f"`{args.docker_image}`"],
        ["Runner", f"`{runner_display}`"],
        ["Shards", f"{received_reports} / {expected_reports} reported"],
        ["Selection", selection_content],
        [
            "实际执行用例",
            (
                f"{totals['total']} total; {totals['passed']} passed; {totals['failed']} failed; "
                f"{totals['errors']} errors; {totals['skipped']} skipped"
            ),
        ],
    ]
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

    # Add shard-level statistics table
    if sorted_shards:
        markdown_lines.extend(["", "## 用例级执行统计"])
        markdown_lines.extend(
            render_table(
                ["Shard", "总用例", "通过", "失败", "错误", "跳过", "Duration"],
                [
                    [
                        f"{row['shard']}",
                        str(row["total"]),
                        str(row["passed"]),
                        str(row["failed"]),
                        str(row["errors"]),
                        str(row.get("skipped", 0)),
                        format_duration(row["duration"]),
                    ]
                    for row in sorted_shards
                ],
            )
        )

        # Build file-level statistics from collected JSONL per-file records
        merged_file_stats = {}
        for rec in all_file_records:
            test_file = rec.get("test_file", "")
            if test_file not in merged_file_stats:
                merged_file_stats[test_file] = {
                    "file": test_file,
                    "total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0,
                    "duration": 0.0, "case_count": len(rec.get("cases", [])),
                    "test_type": rec.get("shard_type", "unknown"),
                }
            fs = merged_file_stats[test_file]
            for c in rec.get("cases", []):
                st = c.get("status", "error")
                fs["total"] += 1
                fs[st] = fs.get(st, 0) + 1
            fs["duration"] += rec.get("duration") or 0.0

        if merged_file_stats:
            # Sort files by total cases descending
            sorted_files = sorted(
                merged_file_stats.values(),
                key=lambda x: (-x["case_count"], x["file"])
            )

            markdown_lines.extend(["", "## 测试文件结果汇总"])

            file_rows = []
            for fs in sorted_files:
                # Calculate fail rate based on executed cases
                failed_total = fs["failed"] + fs["errors"]
                fail_rate = f"{(failed_total / fs['total'] * 100):.1f}%" if fs["total"] > 0 else "0%"
                # Shard info from test_type (each file belongs to one category)
                shard_info = fs.get("test_type", "-")
                file_rows.append([
                    sanitize_markdown_cell(fs["file"]),
                    shard_info,
                    str(fs["case_count"]),
                    str(fs["passed"]),
                    str(fs["failed"]),
                    str(fs["errors"]),
                    str(fs["skipped"]),
                    fail_rate,
                ])

            markdown_lines.extend(
                render_table(
                    ["测试文件", "分片", "规划用例", "通过", "失败", "错误", "跳过", "失败率"],
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

    # Write Markdown report
    output_markdown.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print(f"Generated markdown report: {output_markdown}")

    # Write aggregated JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        # Line 1: global summary
        summary = {
            "shard_type": "all",
            "execution_mode": ", ".join(sorted(execution_modes)) if execution_modes else "file_level_upstream",
            "runner": runner_display,
            "total_files": len(all_file_records),
            "total_cases": totals["total"],
            "passed": totals["passed"],
            "failed": totals["failed"],
            "errors": totals["errors"],
            "skipped": totals["skipped"],
            "shards_reported": f"{received_reports} / {expected_reports}",
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        # Lines 2+: per-file records
        for rec in sorted(all_file_records, key=lambda r: r.get("test_file", "")):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Generated aggregated JSONL: {output_jsonl} ({len(all_file_records)} files)")


if __name__ == "__main__":
    main()
