#!/usr/bin/env python3
"""
Generate a consolidated markdown/json report for the NPU full test workflow.

Output files:
- npu-full-test-summary.json: Lightweight summary with aggregated stats only
- distributed_cases_results_by_file.jsonl: Case-level results grouped by file
- regular_cases_results_by_file.jsonl: Case-level results grouped by file
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import aggregation function from parse_test_results.py
import parse_test_results


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
    parser.add_argument("--output-json", required=True, help="Path to write JSON summary")
    parser.add_argument("--pytorch-version", required=True, help="PyTorch version string")
    parser.add_argument("--torch-npu-whl", required=True, help="torch_npu wheel URL")
    parser.add_argument("--patch-count", default="N/A", help="Applied patch count")
    parser.add_argument("--shard-matrix-json", required=True, help="JSON array of requested shard ids")
    parser.add_argument("--docker-image", default="N/A", help="Docker image used for test execution")
    parser.add_argument("--runner", default="N/A", help="Runner machine type")
    parser.add_argument("--special-reports-root", help="Root directory containing special test report files")
    parser.add_argument("--expected-special-tests-json", default="[]", help="JSON array of expected special test names")
    parser.add_argument("--cases-summary", help="Path to cases_collection_summary.json for file discovery stats")
    parser.add_argument("--cases-by-file-dir", help="Directory containing *_cases_by_file.jsonl files")
    parser.add_argument("--output-html", help="Path to write HTML report (optional, self-contained single file)")
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
    - Type-prefixed: ["dist-1", "reg-2", "custom-1"] -> [("distributed", 1), ("regular", 2), ("custom", 1)]
    - With subtype: ["dist-2card-1", "reg-cpu-1", "reg-npu-2"] ->
        [("distributed_2card", 1), ("regular_cpu", 1), ("regular_npu", 2)]

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
                parts = item.split("-")
                # Formats: "dist-1" (2 parts), "dist-2card-1" (3 parts),
                #          "reg-1" (2 parts), "reg-cpu-1" (3 parts)
                if len(parts) == 2:
                    prefix, num_str = parts
                    shard_num = int(num_str)
                elif len(parts) == 3:
                    prefix, subtype, num_str = parts
                    shard_num = int(num_str)
                else:
                    # No dash, try plain int
                    shard_num = int(item)
                    result.append(("regular", shard_num))
                    continue

                if prefix == "dist":
                    shard_type = f"distributed_{subtype}" if len(parts) == 3 else "distributed"
                elif prefix == "reg":
                    shard_type = f"regular_{subtype}" if len(parts) == 3 else "regular"
                elif prefix == "custom":
                    shard_type = "custom"
                else:
                    continue
                result.append((shard_type, shard_num))
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

    def parse_shard_filename(path: Path, suffix_pattern: str) -> Optional[Tuple[str, int]]:
        """
        Parse shard type and number from filename.

        Filename format: shard_{type}-{number}_{suffix}
        e.g., shard_dist-1_stats.json -> ("distributed", 1)
              shard_reg-1_stats.json -> ("regular", 1)
              shard_custom-1_stats.json -> ("custom", 1)
        """
        stem = path.stem  # filename without extension
        # Match: shard_{type}-{number}_{suffix}  OR  shard_{type}-{sub}-{number}_{suffix}
        # e.g. shard_dist-1_stats.json  → ("distributed", 1)
        #      shard_dist-2card-1_stats.json → ("distributed_2card", 1)
        match = re.match(r"shard_(dist)(?:-(\w+))?-(\d+)_" + suffix_pattern, stem)
        if match:
            type_prefix = match.group(1)      # "dist"
            type_sub = match.group(2)          # "2card" or None
            shard_num = int(match.group(3))
            if type_prefix == "dist":
                if type_sub:
                    return (f"distributed_{type_sub}", shard_num)
                return ("distributed", shard_num)
            return None
        # Match: shard_reg-1_stats.json OR shard_reg-cpu-1_stats.json OR shard_reg-npu-1_stats.json
        match = re.match(r"shard_(reg)(?:-(\w+))?-(\d+)_" + suffix_pattern, stem)
        if match:
            type_sub = match.group(2)          # "cpu", "npu" or None
            shard_num = int(match.group(3))
            if type_sub:
                return (f"regular_{type_sub}", shard_num)
            return ("regular", shard_num)
        match = re.match(r"shard_(custom)-(\d+)_" + suffix_pattern, stem)
        if match:
            shard_num = int(match.group(2))
            return ("custom", shard_num)
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

            # Build shard ID: "dist-1", "dist-2card-1", "reg-2", "reg-cpu-1", "reg-npu-1"
            shard_id = parse_test_results.get_shard_type_prefix(test_type) + f"-{shard_num}"

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
        # Sort by type (dist first) then number; last segment is always the shard num
        file_to_shards[file_path].sort(key=lambda x: (0 if x.startswith("dist") else 1, int(x.rsplit("-", 1)[-1])))

    return file_to_shards


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


def markdown_to_html(markdown_text: str, title: str = "PyTorch NPU Full Test Summary") -> str:
    """Convert the generated markdown report to a self-contained HTML page.

    Handles the subset of markdown produced by this script:
    H1/H2 headings and pipe tables.
    """
    import html as html_lib

    STATUS_CLASSES = {
        "PASSED": "ok",
        "FAILED": "fail",
        "ERROR": "fail",
        "MISSING": "fail",
        "TIMEOUT": "fail",
        "INCOMPLETE": "fail",
        "NO TESTS": "warn",
    }

    def render_cell(cell: str, is_header: bool) -> str:
        # 还原 markdown 单元格中的转义与换行
        text = cell.replace("\\|", "|")
        escaped = html_lib.escape(text, quote=False)
        # sanitize_markdown_cell 用 <br> 表示换行, 转义后还原为真实换行标签
        escaped = escaped.replace("&lt;br&gt;", "<br>")
        # 内联代码 `xxx` 渲染为 <code>xxx</code>
        parts = escaped.split("`")
        escaped = "".join(
            f"<code>{part}</code>" if idx % 2 == 1 else part
            for idx, part in enumerate(parts)
        )
        css = ""
        if not is_header:
            stripped = cell.strip()
            cls = STATUS_CLASSES.get(stripped)
            if cls:
                css = f' class="{cls}"'
            elif stripped.endswith("%") and stripped != "0%" and stripped != "0.0%":
                css = ' class="fail"'
        return f"<td{css}>{escaped}</td>" if not is_header else f"<th>{escaped}</th>"

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html_lib.escape(title)}</title>",
        "<style>",
        "body{font-family:-apple-system,'Segoe UI',Arial,'Microsoft YaHei',sans-serif;"
        "margin:0;padding:24px;background:#f5f6f8;color:#24292f;}",
        "h1{font-size:22px;border-bottom:2px solid #d0d7de;padding-bottom:8px;}",
        "h2{font-size:17px;margin-top:28px;}",
        "table{border-collapse:collapse;margin:12px 0;background:#fff;font-size:13px;max-width:100%;}",
        "th,td{border:1px solid #d0d7de;padding:6px 10px;text-align:left;}",
        "th{background:#f0f3f6;white-space:nowrap;}",
        "tr:nth-child(even) td{background:#fafbfc;}",
        "td.ok{color:#1a7f37;font-weight:600;}",
        "td.fail{color:#cf222e;font-weight:600;}",
        "td.warn{color:#9a6700;font-weight:600;}",
        "footer{margin-top:32px;color:#57606a;font-size:12px;}",
        "</style>",
        "</head>",
        "<body>",
    ]

    lines = markdown_text.splitlines()
    i = 0
    in_table = False

    def close_table():
        nonlocal in_table
        if in_table:
            html_parts.append("</table>")
            in_table = False

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            close_table()
        elif line.startswith("# "):
            close_table()
            html_parts.append(f"<h1>{html_lib.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            close_table()
            html_parts.append(f"<h2>{html_lib.escape(line[3:])}</h2>")
        elif line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # 跳过分隔行 (| --- | --- |)
            if all(set(c) <= set("-: ") and c for c in cells):
                i += 1
                continue
            if not in_table:
                html_parts.append("<table>")
                in_table = True
                is_header = True
            else:
                is_header = False
            tag_cells = "".join(render_cell(c, is_header) for c in cells)
            html_parts.append(f"<tr>{tag_cells}</tr>")
        else:
            close_table()
            html_parts.append(f"<p>{html_lib.escape(line)}</p>")
        i += 1
    close_table()

    html_parts.extend([
        "<footer>Generated by nightly pipeline report generator</footer>",
        "</body>",
        "</html>",
    ])
    return "\n".join(html_parts) + "\n"


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


def load_cases_by_file_jsonl(jsonl_path: Path) -> Tuple[Dict, List[Dict]]:
    """
    Load cases_by_file.jsonl file.

    Returns:
        Tuple of (summary_dict, file_data_list)
        - summary_dict: {"total_file": xxx, "total_cases": xxx}
        - file_data_list: [{"file_path": xxx, "case_count": xxx, "cases": [nodeid1, ...]}, ...]
    """
    if not jsonl_path or not jsonl_path.exists():
        return {}, []

    summary_dict = {}
    file_data_list = []

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if i == 0 and "total_file" in obj:
                    # First line is summary
                    summary_dict = obj
                elif "file_path" in obj:
                    # File data line
                    file_data_list.append(obj)
    except Exception as e:
        print(f"Warning: Failed to load {jsonl_path}: {e}")

    return summary_dict, file_data_list


def build_nodeid_to_case_map(cases_results: Dict) -> Dict[str, Dict]:
    """
    Build a mapping from nodeid to case execution result.

    Args:
        cases_results: Dict from shard_key -> cases_data

    Returns:
        Dict mapping nodeid -> case result dict
    """
    nodeid_to_case = {}
    for shard_key, cases_data in cases_results.items():
        cases_list = cases_data.get("cases", [])
        for case in cases_list:
            nodeid = case.get("nodeid", "")
            if nodeid:
                nodeid_to_case[nodeid] = case
    return nodeid_to_case


def generate_cases_results_jsonl(
    test_type: str,
    file_data_list: List[Dict],
    summary_dict: Dict,
    nodeid_to_case: Dict,
    output_dir: Path,
) -> Path:
    """
    Generate JSONL file with case execution results grouped by file.

    Format:
    Line 1: {"total_file":xxx,"total_cases":xxx}
    Line 2+: {"file_path":"xxx","case_count":xxx,"cases":[{"nodeid":"xxx","status":"passed",...},...]}

    Args:
        test_type: "distributed" or "regular"
        file_data_list: List of file data dicts from *_cases_by_file.jsonl
        summary_dict: Summary dict from *_cases_by_file.jsonl
        nodeid_to_case: Mapping from nodeid to case execution result
        output_dir: Output directory

    Returns:
        Path to generated JSONL file
    """
    output_file = output_dir / f"{test_type}_cases_results_by_file.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        # Line 1: summary (use compact JSON)
        summary_line = json.dumps(summary_dict, separators=(',', ':'))
        f.write(summary_line + '\n')

        # Line 2+: file data with enriched case results
        for file_data in file_data_list:
            file_path = file_data.get("file_path", "")
            nodeids = file_data.get("cases", [])

            # Enrich nodeids with execution results
            enriched_cases = []
            for nodeid in nodeids:
                case_result = nodeid_to_case.get(nodeid, {})
                if case_result:
                    # Case has execution result
                    enriched_cases.append({
                        "nodeid": case_result.get("nodeid", nodeid),
                        "status": case_result.get("status", "unknown"),
                        "duration": case_result.get("duration", 0.0),
                        "returncode": case_result.get("returncode", 0),
                        "message": case_result.get("message", ""),
                        "command": case_result.get("command", ""),
                        "file": case_result.get("file", file_path),
                        "case_idx": case_result.get("case_idx", 0),
                    })
                else:
                    # Case not executed (missing from results)
                    enriched_cases.append({
                        "nodeid": nodeid,
                        "status": "not_executed",
                        "duration": 0.0,
                        "returncode": 0,
                        "message": "",
                        "command": "",
                        "file": file_path,
                        "case_idx": 0,
                    })

            file_line = json.dumps({
                "file_path": file_path,
                "case_count": len(enriched_cases),
                "cases": enriched_cases,
            }, separators=(',', ':'))
            f.write(file_line + '\n')

    print(f"Generated {test_type}_cases_results_by_file.jsonl: {len(file_data_list)} files -> {output_file}")
    return output_file


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

        # Convert shard_type to display prefix ("distributed" -> "dist", "regular" -> "reg", "custom" -> "custom")
        if shard_type == "distributed":
            shard_prefix = "dist"
        elif shard_type == "custom":
            shard_prefix = "custom"
        elif shard_type.startswith("distributed_"):
            # distributed_2card -> dist-2card
            shard_prefix = f"dist-{shard_type[len('distributed_'):]}"
        elif shard_type.startswith("regular_"):
            # regular_cpu -> reg-cpu, regular_npu -> reg-npu
            shard_prefix = f"reg-{shard_type[len('regular_'):]}"
        else:
            shard_prefix = "reg"
        shard_rows.append(
            {
                "shard": f"{shard_prefix}-{shard_num}",  # "dist-1", "reg-1", "reg-cpu-1", or "custom-1"
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

    if any(row["status"] != STATUS_PASSED for row in special_test_rows):
        overall_status = STATUS_FAILED

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
    disabled_filtered_count = 0
    if cases_summary_data:
        planned_total_cases = cases_summary_data.get("total_cases", 0)
        planned_dist_cases = cases_summary_data.get("distributed", {}).get("cases_summary", {}).get("total_cases", 0)
        planned_reg_cases = cases_summary_data.get("regular", {}).get("cases_summary", {}).get("total_cases", 0)
        disabled_filtered_count = cases_summary_data.get("disabled_filtered_count", 0)

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
    if disabled_filtered_count > 0:
        overview_rows.append([
            "黑名单过滤",
            f"{disabled_filtered_count} cases skipped (disabled testcases/methods)",
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

        # Build file-level statistics from jsonl (full file set) + execution results
        file_stats = parse_test_results.aggregate_all_cases_by_file(cases_results)

        # Load all files from jsonl (includes files with 0 cases that weren't executed)
        all_files_from_jsonl = {}
        if args.cases_by_file_dir:
            cases_by_file_dir = Path(args.cases_by_file_dir)
            dist_jsonl_path = cases_by_file_dir / "distributed_cases_by_file.jsonl"
            reg_jsonl_path = cases_by_file_dir / "regular_cases_by_file.jsonl"

            if dist_jsonl_path.exists():
                _, dist_file_data = load_cases_by_file_jsonl(dist_jsonl_path)
                for fd in dist_file_data:
                    file_path = fd.get("file_path", "")
                    all_files_from_jsonl[file_path] = {
                        "file": file_path,
                        "case_count": fd.get("case_count", 0),
                        "test_type": "distributed",
                    }

            if reg_jsonl_path.exists():
                _, reg_file_data = load_cases_by_file_jsonl(reg_jsonl_path)
                for fd in reg_file_data:
                    file_path = fd.get("file_path", "")
                    all_files_from_jsonl[file_path] = {
                        "file": file_path,
                        "case_count": fd.get("case_count", 0),
                        "test_type": "regular",
                    }

        # Merge execution results with full file set
        merged_file_stats = {}
        for file_path, file_info in all_files_from_jsonl.items():
            exec_stats = file_stats.get(file_path, {})
            merged_file_stats[file_path] = {
                "file": file_path,
                "total": exec_stats.get("total", 0),
                "passed": exec_stats.get("passed", 0),
                "failed": exec_stats.get("failed", 0),
                "errors": exec_stats.get("errors", 0),
                "timeout": exec_stats.get("timeout", 0),
                "skipped": exec_stats.get("skipped", 0),
                "duration": exec_stats.get("duration", 0.0),
                "case_count": file_info.get("case_count", 0),  # 规划用例数（可能 > 执行用例数）
                "test_type": file_info.get("test_type", "unknown"),
            }

        # Also add files that were executed but not in jsonl (edge case)
        for file_path, exec_stats in file_stats.items():
            if file_path not in merged_file_stats:
                merged_file_stats[file_path] = {
                    "file": file_path,
                    "total": exec_stats.get("total", 0),
                    "passed": exec_stats.get("passed", 0),
                    "failed": exec_stats.get("failed", 0),
                    "errors": exec_stats.get("errors", 0),
                    "timeout": exec_stats.get("timeout", 0),
                    "skipped": exec_stats.get("skipped", 0),
                    "duration": exec_stats.get("duration", 0.0),
                    "case_count": exec_stats.get("total", 0),
                    "test_type": "unknown",
                }

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
                failed_total = fs["failed"] + fs["errors"] + fs["timeout"]
                fail_rate = f"{(failed_total / fs['total'] * 100):.1f}%" if fs["total"] > 0 else "0%"
                # Get shard info for this file
                file_path = fs["file"]
                # Normalize file path for lookup (remove leading "test/")
                lookup_path = file_path
                if lookup_path.startswith("test/"):
                    lookup_path = lookup_path[5:]
                shards_for_file = file_to_shards_map.get(lookup_path, [])
                # If case_count is 0, no shard executed this file
                shard_info = ", ".join(shards_for_file) if shards_for_file else "-"
                file_rows.append([
                    sanitize_markdown_cell(fs["file"]),
                    shard_info,
                    str(fs["case_count"]),  # 规划用例数
                    str(fs["passed"]),
                    str(fs["failed"]),
                    str(fs["errors"]),
                    str(fs["skipped"]),
                    str(fs["timeout"]),
                    fail_rate,
                ])

            markdown_lines.extend(
                render_table(
                    ["测试文件", "分片", "规划用例", "通过", "失败", "错误", "跳过", "超时", "失败率"],
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
        "disabled_filtered_count": disabled_filtered_count,
        "shards": shard_rows,
    }

    # Add cases collection summary (lightweight metadata only for md rendering)
    if cases_summary_data:
        report_json["cases_collection_summary"] = {
            "total_cases": cases_summary_data.get("total_cases", 0),
            "total_files_scanned": cases_summary_data.get("total_files_scanned", 0),
            "distributed_files": cases_summary_data.get("distributed_files", 0),
            "regular_files": cases_summary_data.get("regular_files", 0),
            "distributed": {
                "total_cases": cases_summary_data.get("distributed", {}).get("cases_summary", {}).get("total_cases", 0),
            },
            "regular": {
                "total_cases": cases_summary_data.get("regular", {}).get("cases_summary", {}).get("total_cases", 0),
            },
        }

    # Generate JSONL files with case-level results grouped by file
    if cases_results and args.cases_by_file_dir:
        cases_by_file_dir = Path(args.cases_by_file_dir)
        output_dir = output_json.parent

        # Build nodeid to case result mapping
        nodeid_to_case = build_nodeid_to_case_map(cases_results)

        # Process distributed cases
        dist_jsonl_path = cases_by_file_dir / "distributed_cases_by_file.jsonl"
        if dist_jsonl_path.exists():
            dist_summary, dist_file_data = load_cases_by_file_jsonl(dist_jsonl_path)
            generate_cases_results_jsonl(
                "distributed",
                dist_file_data,
                dist_summary,
                nodeid_to_case,
                output_dir,
            )

        # Process regular cases
        reg_jsonl_path = cases_by_file_dir / "regular_cases_by_file.jsonl"
        if reg_jsonl_path.exists():
            reg_summary, reg_file_data = load_cases_by_file_jsonl(reg_jsonl_path)
            generate_cases_results_jsonl(
                "regular",
                reg_file_data,
                reg_summary,
                nodeid_to_case,
                output_dir,
            )

    # Add special tests if applicable
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

    if args.output_html:
        output_html = Path(args.output_html)
        html_content = markdown_to_html("\n".join(markdown_lines))
        output_html.write_text(html_content, encoding="utf-8")
        print(f"Generated html report: {output_html}")


if __name__ == "__main__":
    main()
