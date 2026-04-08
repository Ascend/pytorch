#!/usr/bin/env python3
"""
Generate a consolidated markdown/json report for the NPU full test workflow.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Generate consolidated NPU full test report")
    parser.add_argument("--reports-root", required=True, help="Root directory containing shard report files")
    parser.add_argument("--output-markdown", required=True, help="Path to write markdown report")
    parser.add_argument("--output-json", required=True, help="Path to write JSON report")
    parser.add_argument("--pytorch-version", required=True, help="PyTorch version string")
    parser.add_argument("--torch-npu-whl", required=True, help="torch_npu wheel URL")
    parser.add_argument("--patch-count", default="N/A", help="Applied patch count")
    parser.add_argument("--shard-matrix-json", required=True, help="JSON array of requested shard ids")
    return parser.parse_args()


def load_json_file(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_requested_shards(raw: str) -> List[int]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(value, list):
        return []

    result = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(result))


def load_text_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def discover_shard_files(
    reports_root: Path,
) -> Tuple[Dict[int, Path], Dict[int, Path], Dict[int, Path], Dict[int, Path]]:
    stats_files = {}
    info_files = {}
    plan_files = {}
    excluded_files = {}

    for path in reports_root.rglob("shard_*_stats.json"):
        try:
            shard = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        stats_files[shard] = path

    for path in reports_root.rglob("shard_*_info.json"):
        try:
            shard = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        info_files[shard] = path

    for path in reports_root.rglob("shard_*_planned_test_files.txt"):
        try:
            shard = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        plan_files[shard] = path

    for path in reports_root.rglob("shard_*_excluded_test_files.txt"):
        try:
            shard = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        excluded_files[shard] = path

    return stats_files, info_files, plan_files, excluded_files


def get_shard_status(stats: Dict, present: bool) -> str:
    if not present:
        return "MISSING"
    if stats.get("crashed"):
        return "CRASHED"
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
    if any(status_counts[key] > 0 for key in ("CRASHED", "TIMEOUT", "INCOMPLETE", "ERROR", "FAILED")):
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


def build_note(stats: Dict) -> str:
    notes = []
    if stats.get("crash_signal"):
        notes.append(stats["crash_signal"])
    if stats.get("timed_out"):
        notes.append("overall timeout")
    if stats.get("incomplete"):
        notes.append("no junit xml")
    if stats.get("error_message"):
        notes.append(stats["error_message"])
    return "; ".join(notes)


def sanitize_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def format_planned_files_cell(planned_files: List[str]) -> str:
    if not planned_files:
        return "-"
    return "<br>".join(sanitize_markdown_cell(path) for path in planned_files)


def format_scope_list(items: List[str]) -> List[str]:
    if not items:
        return ["- None"]
    return [f"- {sanitize_markdown_cell(item)}" for item in items]


def render_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def main():
    args = parse_args()
    reports_root = Path(args.reports_root)
    output_markdown = Path(args.output_markdown)
    output_json = Path(args.output_json)
    requested_shards = parse_requested_shards(args.shard_matrix_json)

    stats_files, info_files, plan_files, excluded_files = discover_shard_files(reports_root)
    shard_ids = requested_shards or sorted(set(stats_files) | set(info_files))

    status_counts = Counter()
    totals = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0.0,
        "discovered_test_files": 0,
        "planned_files": 0,
    }
    shard_rows = []
    unique_planned_files = set()
    unique_excluded_files = set()
    excluded_dirs = set()

    for shard in shard_ids:
        stats_path = stats_files.get(shard)
        info_path = info_files.get(shard)
        plan_path = plan_files.get(shard)
        excluded_path = excluded_files.get(shard)
        stats = load_json_file(stats_path) if stats_path else {}
        info = load_json_file(info_path) if info_path else {}
        planned_files = load_text_lines(plan_path) if plan_path else []
        excluded_test_files = load_text_lines(excluded_path) if excluded_path else []
        present = bool(stats_path)

        unique_planned_files.update(planned_files)
        unique_excluded_files.update(excluded_test_files)
        excluded_dirs.update(info.get("excluded_dirs", []))

        status = get_shard_status(stats, present)
        status_counts[status] += 1

        totals["total"] += int(stats.get("total", 0))
        totals["passed"] += int(stats.get("passed", 0))
        totals["failed"] += int(stats.get("failed", 0))
        totals["skipped"] += int(stats.get("skipped", 0))
        totals["errors"] += int(stats.get("errors", 0))
        totals["duration"] += float(stats.get("duration", 0.0))
        totals["discovered_test_files"] = max(
            totals["discovered_test_files"], int(info.get("total_files", 0))
        )
        totals["planned_files"] += int(info.get("shard_files", 0))

        shard_rows.append(
            {
                "shard": shard,
                "status": status,
                "total": int(stats.get("total", 0)),
                "passed": int(stats.get("passed", 0)),
                "failed": int(stats.get("failed", 0)),
                "skipped": int(stats.get("skipped", 0)),
                "errors": int(stats.get("errors", 0)),
                "duration": float(stats.get("duration", 0.0)),
                "planned_files": int(info.get("shard_files", 0)),
                "discovered_test_files": int(info.get("total_files", 0)),
                "planned_file_names": planned_files,
                "excluded_test_files": int(info.get("excluded_test_files", 0)),
                "disabled_matched": int(info.get("disabled_count_matched", 0)),
                "disabled_deselected": int(info.get("disabled_count_deselected", 0)),
                "note": build_note(stats),
            }
        )

    overall_status = get_overall_status(status_counts)
    whl_name = Path(args.torch_npu_whl).name
    received_reports = len(stats_files)
    expected_reports = len(shard_ids)
    unique_planned_count = len(unique_planned_files)
    excluded_dirs_list = sorted(excluded_dirs)
    excluded_test_files_list = sorted(unique_excluded_files)
    not_covered_by_requested_shards = max(
        totals["discovered_test_files"] - unique_planned_count,
        0,
    )
    not_covered_display = str(not_covered_by_requested_shards)
    if received_reports < expected_reports:
        not_covered_display = (
            f"{not_covered_by_requested_shards} (based on collected reports only; some shard reports are missing)"
        )

    failed_like = [row for row in shard_rows if row["status"] not in ("PASSED", "NO TESTS")]
    slowest = sorted(shard_rows, key=lambda row: row["duration"], reverse=True)[:20]

    markdown_lines = [
        "# PyTorch NPU Full Test Summary",
        "",
        "## Overview",
    ]
    markdown_lines.extend(
        render_table(
            ["Item", "Value"],
            [
                ["PyTorch", f"`v{args.pytorch_version}`"],
                ["torch_npu", f"`{whl_name}`"],
                ["Patches applied", str(args.patch_count)],
                ["Requested shards", str(expected_reports)],
                ["Reports collected", f"{received_reports} / {expected_reports}"],
                ["Discovered test files", str(totals["discovered_test_files"])],
                ["Planned files in requested shards", str(totals["planned_files"])],
                ["Overall result", overall_status],
            ],
        )
    )
    markdown_lines.extend(["", "## Totals"])
    markdown_lines.extend(
        render_table(
            ["Metric", "Value"],
            [
                ["Total", str(totals["total"])],
                ["Passed", str(totals["passed"])],
                ["Failed", str(totals["failed"])],
                ["Skipped", str(totals["skipped"])],
                ["Errors", str(totals["errors"])],
                ["Discovered test files", str(totals["discovered_test_files"])],
                ["Planned files in requested shards", str(totals["planned_files"])],
                ["Cumulative duration", format_duration(totals["duration"])],
            ],
        )
    )
    markdown_lines.extend(["", "## Execution Scope"])
    markdown_lines.extend(
        render_table(
            ["Item", "Value"],
            [
                ["Unique planned test files in collected reports", str(unique_planned_count)],
                ["Files not covered by requested shard range", not_covered_display],
                ["Excluded directories by code logic", ", ".join(excluded_dirs_list) if excluded_dirs_list else "-"],
                ["Excluded test files by code logic", str(len(excluded_test_files_list))],
            ],
        )
    )
    markdown_lines.extend(["", "### Excluded Directories"])
    markdown_lines.extend(format_scope_list(excluded_dirs_list))
    markdown_lines.extend(["", "### Excluded Test Files"])
    markdown_lines.extend(format_scope_list(excluded_test_files_list))
    markdown_lines.extend(["", "## Shard Status Counts"])
    markdown_lines.extend(
        render_table(
            ["Status", "Shard count"],
            [
                ["PASSED", str(status_counts["PASSED"])],
                ["FAILED", str(status_counts["FAILED"])],
                ["ERROR", str(status_counts["ERROR"])],
                ["CRASHED", str(status_counts["CRASHED"])],
                ["TIMEOUT", str(status_counts["TIMEOUT"])],
                ["INCOMPLETE", str(status_counts["INCOMPLETE"])],
                ["NO TESTS", str(status_counts["NO TESTS"])],
                ["MISSING", str(status_counts["MISSING"])],
            ],
        )
    )
    markdown_lines.extend(["", "## Per-Shard Results"])
    markdown_lines.extend(
        render_table(
            [
                "Shard",
                "Status",
                "Total",
                "Passed",
                "Failed",
                "Skipped",
                "Errors",
                "Duration",
                "Planned Files",
                "Planned File Names",
                "Disabled matched",
                "Note",
            ],
            [
                [
                    str(row["shard"]),
                    row["status"],
                    str(row["total"]),
                    str(row["passed"]),
                    str(row["failed"]),
                    str(row["skipped"]),
                    str(row["errors"]),
                    format_duration(row["duration"]),
                    str(row["planned_files"]),
                    format_planned_files_cell(row["planned_file_names"]),
                    str(row["disabled_matched"]),
                    sanitize_markdown_cell(row["note"] or "-"),
                ]
                for row in sorted(shard_rows, key=lambda row: row["shard"])
            ],
        )
    )
    markdown_lines.extend(["", "## Slowest Shards"])
    markdown_lines.extend(
        render_table(
            ["Shard", "Status", "Duration", "Total", "Failed", "Planned Files"],
            [
                [
                    str(row["shard"]),
                    row["status"],
                    format_duration(row["duration"]),
                    str(row["total"]),
                    str(row["failed"]),
                    str(row["planned_files"]),
                ]
                for row in slowest
            ],
        )
    )

    report_json = {
        "overall_status": overall_status,
        "requested_shards": shard_ids,
        "reports_collected": received_reports,
        "patch_count": args.patch_count,
        "pytorch_version": args.pytorch_version,
        "torch_npu_whl": whl_name,
        "status_counts": dict(status_counts),
        "totals": totals,
        "execution_scope": {
            "unique_planned_test_files": unique_planned_count,
            "files_not_covered_by_requested_shards": not_covered_by_requested_shards,
            "excluded_directories": excluded_dirs_list,
            "excluded_test_files": excluded_test_files_list,
        },
        "shards": shard_rows,
        "failed_like_shards": failed_like,
        "slowest_shards": slowest,
    }

    output_markdown.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    output_json.write_text(json.dumps(report_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Generated markdown report: {output_markdown}")
    print(f"Generated json report: {output_json}")


if __name__ == "__main__":
    main()
