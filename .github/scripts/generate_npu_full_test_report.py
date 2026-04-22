#!/usr/bin/env python3
"""
Generate a consolidated markdown/json report for the NPU full test workflow.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional


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
    return parser.parse_args()


def load_json_file(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_junit_xml_testsuites(xml_path: Path) -> List[Dict]:
    """
    Parse JUnit XML file and extract per-testsuite statistics.

    Each testsuite represents a test file with its own stats:
    - name: test file name
    - tests: total test cases
    - failures: failed test cases
    - errors: error test cases
    - skipped: skipped test cases
    - time: execution time in seconds

    Returns a list of testsuite statistics.
    """
    testsuites = []

    if not xml_path.exists():
        return testsuites

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Handle both <testsuites> and <testsuite> as root
        if root.tag == "testsuites":
            for testsuite in root.findall("testsuite"):
                stats = parse_testsuite_element(testsuite)
                if stats:
                    testsuites.append(stats)
        elif root.tag == "testsuite":
            stats = parse_testsuite_element(root)
            if stats:
                testsuites.append(stats)

    except ET.ParseError as e:
        print(f"Warning: Failed to parse XML {xml_path}: {e}")
    except Exception as e:
        print(f"Warning: Error reading XML {xml_path}: {e}")

    return testsuites


def parse_testsuite_element(testsuite: ET.Element) -> Optional[Dict]:
    """Parse a single testsuite element and return its statistics."""
    try:
        name = testsuite.get("name", "unknown")
        tests = int(testsuite.get("tests", 0))
        failures = int(testsuite.get("failures", 0))
        errors = int(testsuite.get("errors", 0))
        skipped = int(testsuite.get("skipped", 0))
        time = float(testsuite.get("time", 0.0))
        passed = tests - failures - errors - skipped

        return {
            "name": name,
            "tests": tests,
            "passed": passed,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "time": time,
        }
    except (ValueError, TypeError):
        return None


def extract_test_identifier(test_path: str) -> str:
    """
    Extract a test identifier from a test file path.

    Converts paths like:
    - "test/distributed/_composable/fsdp/test_fully_shard_autograd.py"
    To:
    - "distributed._composable.fsdp.test_fully_shard_autograd"

    This matches the testsuite naming convention used by pytest/run_test.py.
    """
    # Remove 'test/' prefix if present
    path = test_path
    if path.startswith("test/"):
        path = path[5:]
    # Remove '.py' suffix
    if path.endswith(".py"):
        path = path[:-3]
    # Convert path separators to dots
    path = path.replace("/", ".").replace("\\", ".")
    return path


def aggregate_testsuite_stats_for_shard(
    reports_root: Path,
    shard_type: str,
    shard: int,
    planned_files: List[str],
    missing_files_list: List[str] = None
) -> List[Dict]:
    """
    Aggregate all testsuite statistics for a specific shard.

    The test execution generates XML files named `shard_{type}-{shard}_pytest*.xml`.
    Each XML file contains testcases with `file` attribute indicating the test file.

    Args:
        reports_root: Root directory containing all merged report files
        shard_type: Shard type ("distributed" or "regular")
        shard: Shard number to aggregate for
        planned_files: List of test file paths planned for this shard
        missing_files_list: List of test file paths that crashed and didn't generate XML

    Returns:
        List of testsuite statistics for tests belonging to this shard.
        Missing files are included with status="MISSING" and tests=0.
    """
    if missing_files_list is None:
        missing_files_list = []

    all_testsuites = {}
    # Map from test identifier -> aggregated stats

    # Build set of test identifiers from planned files
    planned_identifiers = set()
    for planned in planned_files:
        identifier = extract_test_identifier(planned)
        if identifier:
            planned_identifiers.add(identifier)

    # Also include just the test file names for simpler matching
    planned_test_names = set()
    for planned in planned_files:
        name = Path(planned).name.replace(".py", "")
        planned_test_names.add(name)

    print(f"DEBUG: planned_files count={len(planned_files)}, planned_identifiers count={len(planned_identifiers)}")
    if planned_identifiers:
        print(f"DEBUG: First 3 planned_identifiers: {list(planned_identifiers)[:3]}")

    # Convert shard_type to file prefix ("distributed" -> "dist", "regular" -> "reg")
    type_prefix = "dist" if shard_type == "distributed" else "reg"

    # Debug: List all files in reports_root
    print(f"DEBUG aggregate_testsuite_stats_for_shard: shard_type={shard_type}, shard={shard}")
    print(f"DEBUG: reports_root={reports_root}, exists={reports_root.exists()}")
    if reports_root.exists():
        all_xml_files = list(reports_root.rglob("*.xml"))
        print(f"DEBUG: Total XML files in reports_root (rglob): {len(all_xml_files)}")
        matching_xml_files = list(reports_root.rglob(f"shard_{type_prefix}-{shard}_pytest*.xml"))
        print(f"DEBUG: Matching XML files for shard_{type_prefix}-{shard}_pytest*.xml (rglob): {len(matching_xml_files)}")
        for xf in matching_xml_files[:5]:
            print(f"DEBUG:   - {xf.relative_to(reports_root)}")

    # Find all XML files for this shard: shard_{type}-{shard}_pytest*.xml
    # Use rglob to search recursively (files may be in subdirectories due to artifact merge)
    for xml_path in reports_root.rglob(f"shard_{type_prefix}-{shard}_pytest*.xml"):
        # Parse testcase elements and aggregate by file attribute
        test_file_stats = aggregate_testcases_by_file(xml_path, planned_identifiers, planned_test_names)
        for test_id, stats in test_file_stats.items():
            if test_id in all_testsuites:
                # Merge with existing stats
                existing = all_testsuites[test_id]
                existing["tests"] += stats["tests"]
                existing["passed"] += stats["passed"]
                existing["failures"] += stats["failures"]
                existing["errors"] += stats["errors"]
                existing["skipped"] += stats["skipped"]
                existing["time"] += stats["time"]
            else:
                all_testsuites[test_id] = stats

    # Also check nested directories for Phase 1 style XMLs (run_test.py output)
    phase1_patterns = [
        "junit",
        "pytorch-test-src/test/test-reports/python-pytest",
    ]

    for phase1_pattern in phase1_patterns:
        phase1_base = reports_root / phase1_pattern
        if not phase1_base.exists():
            continue

        for test_dir in phase1_base.iterdir():
            if not test_dir.is_dir():
                continue
            test_identifier = test_dir.name
            matched = False
            for planned_id in planned_identifiers:
                if test_identifier == planned_id or test_identifier.startswith(planned_id) or planned_id.startswith(test_identifier):
                    matched = True
                    break
            if not matched:
                for test_name in planned_test_names:
                    if test_identifier.endswith(test_name) or test_name in test_identifier:
                        matched = True
                        break
            if not matched:
                continue

            if test_identifier not in all_testsuites:
                all_testsuites[test_identifier] = {
                    "name": test_identifier,
                    "tests": 0,
                    "passed": 0,
                    "failures": 0,
                    "errors": 0,
                    "skipped": 0,
                    "time": 0.0,
                }
            aggregated = all_testsuites[test_identifier]

            for xml_file in test_dir.glob("*.xml"):
                testsuites = parse_junit_xml_testsuites(xml_file)
                for ts in testsuites:
                    aggregated["tests"] += ts.get("tests", 0)
                    aggregated["passed"] += ts.get("passed", 0)
                    aggregated["failures"] += ts.get("failures", 0)
                    aggregated["errors"] += ts.get("errors", 0)
                    aggregated["skipped"] += ts.get("skipped", 0)
                    aggregated["time"] += ts.get("time", 0.0)

    # Add missing files (crashed without generating XML) to the result
    # These files show as "MISSING" in the test file details
    for missing_file in missing_files_list:
        missing_identifier = extract_test_identifier(missing_file)
        if missing_identifier and missing_identifier not in all_testsuites:
            all_testsuites[missing_identifier] = {
                "name": missing_identifier,
                "tests": 0,
                "passed": 0,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "time": 0.0,
                "status": "MISSING",  # Special status for crashed files
            }

    # Convert to list and sort by name
    result = list(all_testsuites.values())
    result.sort(key=lambda x: x["name"])

    return result


def aggregate_testcases_by_file(xml_path: Path, planned_identifiers: set, planned_test_names: set) -> Dict[str, Dict]:
    """
    Parse XML file and aggregate testcase statistics by file attribute.

    Used for XMLs where testsuite name is generic "pytest".
    If planned_identifiers is empty, accept all testcases.
    """
    result = {}
    debug_count = 0

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find all testcase elements
        testcases = root.findall(".//testcase")
        print(f"DEBUG aggregate_testcases_by_file: {xml_path.name}, testcases={len(testcases)}, planned_ids={len(planned_identifiers)}")

        for testcase in testcases:
            file_attr = testcase.get("file", "")
            classname_attr = testcase.get("classname", "")

            # Extract test identifier from file attribute or classname
            test_identifier = None

            if file_attr:
                # e.g., "distributed/fsdp/test_fsdp_sharded_grad_scaler.py"
                test_identifier = extract_test_identifier("test/" + file_attr) if not file_attr.startswith("test/") else extract_test_identifier(file_attr)
                if debug_count < 3:
                    print(f"DEBUG: file_attr='{file_attr}' -> test_identifier='{test_identifier}'")
                    debug_count += 1
            elif classname_attr:
                # classname format: "test.distributed._composable.fsdp.test_fully_shard_comm.TestFullyShardCollectiveOps"
                # The last part is the class name, need to extract the module path
                # e.g., extract "test.distributed._composable.fsdp.test_fully_shard_comm" (module name)
                parts = classname_attr.split(".")
                if len(parts) > 1:
                    # Remove the last part (class name like TestFullyShardCollectiveOps)
                    # Keep everything before the class name
                    module_parts = parts[:-1]
                    classname_attr = ".".join(module_parts)
                # Convert to match planned_identifiers format (dot-separated, no test/ prefix)
                # planned_identifiers format: "distributed._composable.fsdp.test_fully_shard_comm"
                test_identifier = classname_attr
                # Remove 'test.' prefix if present to match planned_identifiers
                if test_identifier.startswith("test."):
                    test_identifier = test_identifier[5:]
                if debug_count < 3:
                    print(f"DEBUG: classname_attr='{classname_attr}' -> test_identifier='{test_identifier}'")
                    debug_count += 1

            if not test_identifier:
                continue

            # If planned_identifiers is empty, accept all testcases
            # Otherwise, check if this test belongs to planned files
            if planned_identifiers or planned_test_names:
                matched = False
                for planned_id in planned_identifiers:
                    if test_identifier == planned_id or test_identifier.startswith(planned_id) or planned_id.startswith(test_identifier):
                        matched = True
                        break
                if not matched:
                    for test_name in planned_test_names:
                        if test_identifier.endswith(test_name) or test_name in test_identifier:
                            matched = True
                            break
                if not matched:
                    continue

            # Initialize stats for this test file
            if test_identifier not in result:
                result[test_identifier] = {
                    "name": test_identifier,
                    "tests": 0,
                    "passed": 0,
                    "failures": 0,
                    "errors": 0,
                    "skipped": 0,
                    "time": 0.0,
                }

            # Count testcase
            stats = result[test_identifier]
            stats["tests"] += 1

            # Determine outcome
            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")

            if failure is not None:
                stats["failures"] += 1
            elif error is not None:
                stats["errors"] += 1
            elif skipped is not None:
                stats["skipped"] += 1
            else:
                stats["passed"] += 1

            # Add time
            time_str = testcase.get("time", "0")
            try:
                stats["time"] += float(time_str)
            except ValueError:
                pass

    except ET.ParseError as e:
        print(f"Warning: Failed to parse XML {xml_path}: {e}")
    except Exception as e:
        print(f"Warning: Error reading XML {xml_path}: {e}")

    return result


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


def get_selected_test_entries(info: Dict) -> int:
    return get_int_value(info, "selected_test_entries", "upstream_selected_tests")


def get_selected_test_files(info: Dict) -> int:
    return get_int_value(info, "selected_test_files", "upstream_selected_file_tests")


def get_path_filtered_out_files(info: Dict) -> int:
    return get_int_value(info, "path_filtered_out_files", "excluded_test_files")


def get_unhandled_special_tests(info: Dict) -> int:
    return get_int_value(info, "unhandled_special_tests", "upstream_unhandled_tests")


def discover_shard_files(
    reports_root: Path,
) -> Tuple[
    Dict[Tuple[str, int], Path],  # stats_files
    Dict[Tuple[str, int], Path],  # info_files
    Dict[Tuple[str, int], Path],  # plan_files
    Dict[Tuple[str, int], Path],  # excluded_files
    Dict[Tuple[str, int], Path],  # unhandled_files
    Dict[Tuple[str, int], Path],  # xml_files
    Dict[Tuple[str, int], Path],  # missing_files
]:
    """
    Discover all shard report files in the reports directory.

    Returns dicts keyed by (shard_type, shard_number) tuples.

    File name format: shard_{type}-{number}_{suffix}
    Examples:
    - shard_dist-1_stats.json
    - shard_reg-1_info.json
    """
    stats_files = {}
    info_files = {}
    plan_files = {}
    excluded_files = {}
    unhandled_files = {}
    xml_files = {}
    missing_files = {}

    def parse_shard_filename(path: Path, suffix_pattern: str) -> Tuple[str, int]:
        """
        Parse shard type and number from filename.

        Filename format: shard_{type}-{number}_{suffix}
        e.g., shard_dist-1_stats.json -> ("distributed", 1)
        e.g., shard_reg-2_planned_test_files.txt -> ("regular", 2)
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

    for path in reports_root.rglob("shard_*_planned_test_files.txt"):
        key = parse_shard_filename(path, "planned_test_files")
        if key:
            plan_files[key] = path

    for path in reports_root.rglob("shard_*_excluded_test_files.txt"):
        key = parse_shard_filename(path, "excluded_test_files")
        if key:
            excluded_files[key] = path

    for path in reports_root.rglob("shard_*_unhandled_upstream_tests.txt"):
        key = parse_shard_filename(path, "unhandled_upstream_tests")
        if key:
            unhandled_files[key] = path

    # Discover XML files for per-test-file statistics
    for path in reports_root.rglob("shard_*_pytest*.xml"):
        # XML filename: shard_{type}-{number}_pytest{suffix}.xml
        stem = path.stem
        match = re.match(r"shard_(dist|reg)-(\d+)_pytest", stem)
        if match:
            type_prefix = match.group(1)
            shard_num = int(match.group(2))
            if type_prefix == "dist":
                key = ("distributed", shard_num)
            elif type_prefix == "reg":
                key = ("regular", shard_num)
            xml_files[key] = path

    # Discover missing files list (files that crashed and didn't generate XML)
    for path in reports_root.rglob("shard_*_missing_files.txt"):
        key = parse_shard_filename(path, "missing_files")
        if key:
            missing_files[key] = path

    return stats_files, info_files, plan_files, excluded_files, unhandled_files, xml_files, missing_files


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


def format_testsuite_detail(stats: Dict) -> str:
    """
    Format a single testsuite's stats for display.

    Format: "test_file.py: 5 passed, 2 failed, 1 error, 0 skipped, 3.2s"
    Or for missing files: "test_file.py: MISSING (crashed, no report)"
    """
    name = sanitize_markdown_cell(stats.get("name", "unknown"))

    # Check for MISSING status (file crashed without generating report)
    if stats.get("status") == "MISSING":
        return f"{name}: MISSING (crashed, no report)"
    passed = stats.get("passed", 0)
    failures = stats.get("failures", 0)
    errors = stats.get("errors", 0)
    skipped = stats.get("skipped", 0)
    time = stats.get("time", 0.0)

    # Build stats parts (comma-separated)
    stats_parts = []
    if passed > 0:
        stats_parts.append(f"{passed} passed")
    if failures > 0:
        stats_parts.append(f"{failures} failed")
    if errors > 0:
        stats_parts.append(f"{errors} error")
    if skipped > 0:
        stats_parts.append(f"{skipped} skipped")
    stats_parts.append(format_duration_short(time))

    # Format: "name: stats1, stats2, ..."
    stats_str = ", ".join(stats_parts)
    return f"{name}: {stats_str}"


def format_duration_short(seconds: float) -> str:
    """Format duration in a compact form for testsuite display."""
    seconds = float(seconds)
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.0f}s"
    return f"{seconds:.1f}s"


def format_testsuite_details_cell(testsuites: List[Dict]) -> str:
    """
    Format all testsuite stats for a shard into a single cell.

    Each testsuite is displayed on a separate line with its stats.
    """
    if not testsuites:
        return "-"

    lines = []
    for ts in testsuites:
        lines.append(format_testsuite_detail(ts))

    return "<br>".join(lines)


def format_summary_note(note: str) -> str:
    cleaned = (note or "").strip()
    if not cleaned or cleaned == "pytest exited with code 1":
        return "-"
    return sanitize_markdown_cell(cleaned)


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

    stats_files, info_files, plan_files, excluded_files, unhandled_files, xml_files, missing_files_paths = discover_shard_files(reports_root)
    special_test_files = discover_special_test_files(special_reports_root)
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
        "selected_test_entries": 0,
        "selected_test_files": 0,
        "path_filtered_out_files": 0,
        "planned_files": 0,
        "junit_generated_shards": 0,
        "junit_xml_files": 0,
        "zero_item_test_files": 0,
        "startup_failures": 0,
        "import_failures": 0,
        "test_failures": 0,
        "missing_files": 0,
    }
    shard_rows = []
    unique_planned_files = set()
    unique_excluded_files = set()
    unique_unhandled_tests = set()
    unique_missing_files = set()
    selection_modes = set()

    for shard_type, shard_num in shard_ids:
        shard_key = (shard_type, shard_num)
        stats_path = stats_files.get(shard_key)
        info_path = info_files.get(shard_key)
        plan_path = plan_files.get(shard_key)
        excluded_path = excluded_files.get(shard_key)
        unhandled_path = unhandled_files.get(shard_key)
        missing_path = missing_files_paths.get(shard_key)
        stats = load_json_file(stats_path) if stats_path else {}
        info = load_json_file(info_path) if info_path else {}
        selected_test_entries = get_selected_test_entries(info)
        selected_test_files = get_selected_test_files(info)
        path_filtered_out_files = get_path_filtered_out_files(info)
        unhandled_special_tests = get_unhandled_special_tests(info)
        planned_files = load_text_lines(plan_path) if plan_path else []
        excluded_test_files = load_text_lines(excluded_path) if excluded_path else []
        unhandled_tests = load_text_lines(unhandled_path) if unhandled_path else []
        missing_files_list = load_text_lines(missing_path) if missing_path else []
        present = bool(stats_path)

        # Parse ALL XML files to get per-test-file statistics
        # This includes Phase 1 (run_test.py) and Phase 2 (pytest fallback) results
        # Filter by planned test files to ensure we only include tests for this shard
        # Include missing files that crashed without generating reports
        testsuite_stats = aggregate_testsuite_stats_for_shard(reports_root, shard_type, shard_num, planned_files, missing_files_list)

        # If testsuite_stats has entries, aggregate their totals and override incomplete status
        has_phase1_xmls = len(testsuite_stats) > 0
        if has_phase1_xmls:
            # Aggregate stats from Phase 1 XMLs
            xml_totals = {
                "tests": 0,
                "passed": 0,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "time": 0.0,
            }
            for ts in testsuite_stats:
                xml_totals["tests"] += ts.get("tests", 0)
                xml_totals["passed"] += ts.get("passed", 0)
                xml_totals["failures"] += ts.get("failures", 0)
                xml_totals["errors"] += ts.get("errors", 0)
                xml_totals["skipped"] += ts.get("skipped", 0)
                xml_totals["time"] += ts.get("time", 0.0)

            # Use XML data to fill stats if:
            # 1. stats.json doesn't exist (stats is empty) but we have XML data
            # 2. stats.json exists but is incomplete and we have XML data to override
            # This ensures per-file isolation mode shards get correct totals even without stats.json
            if xml_totals["tests"] > 0:
                # Always fill stats from XML if stats is empty or incomplete
                if not stats or stats.get("incomplete"):
                    stats["incomplete"] = False
                    stats["total"] = xml_totals["tests"]
                    stats["passed"] = xml_totals["passed"]
                    stats["failed"] = xml_totals["failures"]
                    stats["errors"] = xml_totals["errors"]
                    stats["skipped"] = xml_totals["skipped"]
                    stats["duration"] = xml_totals["time"]
                    # Mark as present if we have XML data (even without stats.json)
                    if not present:
                        present = True

        unique_planned_files.update(planned_files)
        unique_excluded_files.update(excluded_test_files)
        unique_unhandled_tests.update(unhandled_tests)
        unique_missing_files.update(missing_files_list)
        if info.get("selection_mode"):
            selection_modes.add(str(info.get("selection_mode")))

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
        totals["selected_test_entries"] = max(totals["selected_test_entries"], selected_test_entries)
        totals["selected_test_files"] = max(totals["selected_test_files"], selected_test_files)
        totals["path_filtered_out_files"] = max(totals["path_filtered_out_files"], path_filtered_out_files)
        totals["planned_files"] += int(info.get("shard_files", 0))
        totals["junit_generated_shards"] += 1 if info.get("junit_generated") else 0
        totals["junit_xml_files"] += int(info.get("junit_xml_files", 0) or stats.get("junit_xml_files", 0))
        totals["zero_item_test_files"] += int(info.get("zero_item_test_files", 0) or stats.get("zero_item_test_files", 0))
        totals["startup_failures"] += int(info.get("startup_failures", 0) or stats.get("startup_failures", 0))
        totals["import_failures"] += int(info.get("import_failures", 0) or stats.get("import_failures", 0))
        totals["test_failures"] += int(info.get("test_failures", 0) or stats.get("test_failures", 0))
        totals["missing_files"] += len(missing_files_list)

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
                "duration": float(stats.get("duration", 0.0)),
                "planned_files": int(info.get("shard_files", 0)),
                "discovered_test_files": int(info.get("total_files", 0)),
                "selected_test_entries": selected_test_entries,
                "selected_test_files": selected_test_files,
                "unhandled_special_tests": unhandled_special_tests,
                "planned_file_names": planned_files,
                "path_filtered_out_files": path_filtered_out_files,
                "disabled_matched": int(info.get("disabled_count_matched", 0)),
                "disabled_deselected": int(info.get("disabled_count_deselected", 0)),
                "junit_generated": bool(info.get("junit_generated", stats.get("junit_generated", False))),
                "junit_xml_files": int(info.get("junit_xml_files", stats.get("junit_xml_files", 0))),
                "zero_item_test_files": int(info.get("zero_item_test_files", stats.get("zero_item_test_files", 0))),
                "startup_failures": int(info.get("startup_failures", stats.get("startup_failures", 0))),
                "import_failures": int(info.get("import_failures", stats.get("import_failures", 0))),
                "test_failures": int(info.get("test_failures", stats.get("test_failures", 0))),
                "note": build_note(stats),
                "testsuite_stats": testsuite_stats,  # Per-test-file statistics
            }
        )

    overall_status = get_overall_status(status_counts)
    whl_name = Path(args.torch_npu_whl).name
    received_reports = len(stats_files)
    expected_reports = len(shard_ids)
    unique_planned_count = len(unique_planned_files)
    excluded_test_files_list = sorted(unique_excluded_files)
    unhandled_tests_list = sorted(unique_unhandled_tests)
    not_covered_by_requested_shards = max(
        totals["selected_test_files"] - unique_planned_count,
        0,
    )
    selection_mode_display = ", ".join(sorted(selection_modes)) if selection_modes else "-"
    include_selected_entries = totals["selected_test_entries"] > 0
    include_unhandled_tests = bool(unhandled_tests_list)

    # Show all shards in the detail table
    sorted_shards = sorted(shard_rows, key=lambda row: (row["shard_type"], row["shard_num"]))
    slowest = sorted(shard_rows, key=lambda row: row["duration"], reverse=True)[:20]
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

    overview_rows = [
        ["Overall result", overall_status],
        ["PyTorch", f"`v{args.pytorch_version}`"],
        ["torch_npu", f"`{whl_name}`"],
        ["Patches applied", str(args.patch_count)],
        ["Docker image", f"`{args.docker_image}`"],
        ["Runner", f"`{args.runner}`"],
        ["Shards", f"{received_reports} / {expected_reports} reported"],
        [
            "Selection",
            (
                f"{selection_mode_display}; "
                f"{totals['selected_test_files']} selected, "
                f"{totals['path_filtered_out_files']} filtered out"
            ),
        ],
        [
            "Tests",
            (
                f"{totals['total']} total; {totals['passed']} passed; {totals['failed']} failed; "
                f"{totals['skipped']} skipped; {totals['errors']} errors"
            ),
        ],
        ["JUnit", f"{totals['junit_generated_shards']} shards, {totals['junit_xml_files']} xml files"],
        ["Duration", format_duration(totals["duration"])],
    ]
    if include_selected_entries:
        overview_rows.insert(9, ["Selected test entries", str(totals["selected_test_entries"])])
    if totals["missing_files"] > 0:
        overview_rows.append(["Missing files", f"{totals['missing_files']} crashed without report"])
    if include_special_tests:
        overview_rows.append(["Special tests expected", str(len(special_test_names))])

    shard_status_rows = [
        [status, str(status_counts[status])]
        for status in ("CRASHED", "ERROR", "FAILED", "TIMEOUT", "INCOMPLETE", "MISSING", "NO TESTS", "PASSED")
        if status_counts[status] > 0
    ]

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
    markdown_lines.extend(["", "## Shard Status"])
    markdown_lines.extend(
        render_table(
            ["Status", "Shard count"],
            shard_status_rows or [["PASSED", "0"]],
        )
    )
    # Show all shards in detail table
    markdown_lines.extend(["", "## 分片任务详情"])
    markdown_lines.extend(
        render_table(
            ["Shard", "Status", "总用例数", "通过用例数", "Failed", "Errors", "Duration", "测试文件详情", "Note"],
            [
                [
                    str(row["shard"]),
                    row["status"],
                    str(row["total"]),
                    str(row["passed"]),
                    str(row["failed"]),
                    str(row["errors"]),
                    format_duration(row["duration"]),
                    format_testsuite_details_cell(row["testsuite_stats"]),
                    format_summary_note(row["note"]),
                ]
                for row in sorted_shards
            ],
        )
    )
    if include_unhandled_tests:
        markdown_lines.extend(["", "## Unhandled Special Tests"])
        markdown_lines.extend(format_scope_list(unhandled_tests_list))
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
        "execution_scope": {
            "selection_mode": sorted(selection_modes),
            "selected_test_entries": totals["selected_test_entries"],
            "selected_test_files": totals["selected_test_files"],
            "path_filtered_out_files": totals["path_filtered_out_files"],
            "unique_planned_test_files": unique_planned_count,
            "files_not_covered_by_requested_shards": not_covered_by_requested_shards,
            "excluded_test_files": excluded_test_files_list,
            "unhandled_special_tests": unhandled_tests_list,
            "missing_files": sorted(unique_missing_files),
        },
        "failure_breakdown": {
            "startup_failures": totals["startup_failures"],
            "import_failures": totals["import_failures"],
            "test_failures": totals["test_failures"],
            "missing_files": totals["missing_files"],
        },
        "shards": shard_rows,
        "failed_shards": [row for row in shard_rows if row["status"] not in ("PASSED", "NO TESTS")],
        "slowest_shards": slowest,
    }

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
