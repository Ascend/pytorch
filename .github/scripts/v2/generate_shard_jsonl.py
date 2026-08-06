#!/usr/bin/env python3
"""
Generate shard_{prefix}-{shard}_cases.jsonl from JUnit XMLs + run_test.py stderr.

Replaces parse_junit_xmls.py.  Outputs JSONL (one JSON object per line):

    Line 1 — shard summary:
      {"shard":1,"shard_type":"core","execution_mode":"file_level_upstream",
       "runner":"linux-aarch64-a3-8","total_files":3,"total_cases":480,
       "passed":400,"failed":12,"errors":8,"skipped":60}

    Line 2+ — per-file records:
      {"test_file":"test/nn/test_convolution.py","duration":150.0,"return_code":0,
       "message":"","cases":[{...}]}
      {"test_file":"test/nn/test_broken.py","duration":null,"return_code":-11,
       "message":"SIGSEGV: segmentation fault","cases":[]}

File-level status is derived from two sources:
  - JUnit XML: per-case results (passed/failed/error/skipped)
  - run_test.py stderr: per-file timing + exit code + crash signals
    (Finished {test} ... took X.XXmin, {test} failed!, etc.)

Files that appear in stderr but have no JUnit XML are crash files:
return_code is set from the signal name (e.g., -11 for SIGSEGV), message
contains the error description, and cases is [].

Usage:
    python generate_shard_jsonl.py \
        --category core --shard 1 \
        --expected-files /tmp/files.txt \
        --execution-log /tmp/test_npu_core_1.log \
        --reports-dir test-reports
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Map category name → short prefix
_PREFIX_MAP = {
    "core": "core",
    "tensor": "tensor",
    "distributed": "dist",
    "graph": "graph",
    "others": "others",
}

# Signal name → negative return_code mapping
_SIGNAL_MAP = {
    "SIGHUP": -1, "SIGINT": -2, "SIGQUIT": -3, "SIGILL": -4,
    "SIGTRAP": -5, "SIGABRT": -6, "SIGBUS": -7, "SIGFPE": -8,
    "SIGKILL": -9, "SIGUSR1": -10, "SIGSEGV": -11, "SIGUSR2": -12,
    "SIGPIPE": -13, "SIGALRM": -14, "SIGTERM": -15, "SIGSTKFLT": -16,
    "SIGXCPU": -24, "SIGXFSZ": -25, "SIGSYS": -31,
}


# ==============================================================================
# Stderr Parsing
# ==============================================================================


def parse_execution_log(log_path: str):
    """Parse run_test.py stderr to extract per-file timing, exit codes, and errors.

    Returns:
        file_info: dict mapping normalized test file path → {
            "duration": float | None,   # seconds, None if unknown
            "return_code": int | None,  # exit code, None if unknown
            "message": str,             # empty if OK
        }
    """
    file_info = {}

    if not log_path or not os.path.isfile(log_path):
        return file_info

    with open(log_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Pattern 1: Finished {test} ... took X.XXmin
    finished_re = re.compile(
        r"Finished\s+(\S+)\s+\.\.\.\s+.*?took\s+([\d.]+)\s*min"
    )
    for m in finished_re.finditer(content):
        test_name = _match_stderr_to_expected(m.group(1))
        duration_min = float(m.group(2))
        if test_name not in file_info:
            file_info[test_name] = {"duration": None, "return_code": None, "message": ""}
        file_info[test_name]["duration"] = duration_min * 60.0  # min → seconds

    # Pattern 2: {test} failed! [Received signal: SIG*]
    # Lines look like: "test/nn/test_convolution failed!"
    # or: "test/nn/test_convolution failed! Received signal: SIGSEGV"
    failed_re = re.compile(
        r"^(\S+)\s+failed!\s*(?:Received signal:\s*(\S+))?",
        re.MULTILINE,
    )
    for m in failed_re.finditer(content):
        test_name = _match_stderr_to_expected(m.group(1))
        signal_name = m.group(2)
        if test_name not in file_info:
            file_info[test_name] = {"duration": None, "return_code": None, "message": ""}
        if signal_name:
            rc = _SIGNAL_MAP.get(signal_name.upper(), -1)
            file_info[test_name]["return_code"] = rc
            file_info[test_name]["message"] = f"{signal_name}: process killed by signal"
        else:
            file_info[test_name]["return_code"] = 1
            file_info[test_name]["message"] = file_info[test_name].get("message", "")

    # Pattern 3: FAILED CONSISTENTLY: {test}
    consistent_re = re.compile(r"FAILED\s+CONSISTENTLY\s*:\s*(\S+)")
    for m in consistent_re.finditer(content):
        test_name = _match_stderr_to_expected(m.group(1))
        if test_name not in file_info:
            file_info[test_name] = {"duration": None, "return_code": None, "message": ""}
        if file_info[test_name]["return_code"] is None:
            file_info[test_name]["return_code"] = 1

    return file_info


_EXPECTED_FILES_SET = set()


def _match_stderr_to_expected(raw: str) -> str | None:
    """Match a run_test.py stderr test name to an expected file path.

    run_test.py may print names in module form (``test_nn``) or path form
    (``test/nn/test_convolution``).  Expected files from shard_test_files.py are
    always in path form with ``.py``: ``test/nn/test_convolution.py``.

    Matching strategy (first match wins):
      1. Exact match (with or without ``.py``)
      2. Match by last path component (handles module names like ``test_nn``)
    """
    raw = raw.strip()
    pool = _EXPECTED_FILES_SET
    if not pool:
        return None

    # 1) Exact match
    if raw in pool:
        return raw
    if raw + ".py" in pool:
        return raw + ".py"
    if raw.endswith(".py") and raw[:-3] in pool:
        return raw[:-3]

    # 2) Match by trailing filename: "test_nn" ↔ "test/nn/test_nn.py"
    for expected in pool:
        base = expected.rsplit("/", 1)[-1] if "/" in expected else expected  # "test_nn.py"
        if base == raw + ".py" or base == raw:
            return expected
        if base.endswith(".py") and base[:-3] == raw:
            return expected

    return None


# ==============================================================================
# JUnit XML Parsing
# ==============================================================================


def find_xml_files(reports_dir: Path):
    """Find all JUnit XML files under *reports_dir*."""
    xml_files = []
    junit_dir = reports_dir / "junit_xmls"
    if junit_dir.is_dir():
        xml_files.extend(sorted(junit_dir.glob("*.xml")))
    xml_files.extend(sorted(reports_dir.glob("*.xml")))
    return xml_files


def parse_xml_testcases(xml_files):
    """Parse JUnit XML files, grouping cases by test file.

    Returns:
        files_cases: dict mapping test_file → list of case dicts
    """
    files_cases = {}
    seen_nodeids = set()

    for xml_path in xml_files:
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
        except ET.ParseError:
            print(f"  Warning: failed to parse {xml_path.name}", file=sys.stderr)
            continue

        for tc in root.iter("testcase"):
            classname = tc.get("classname", "")
            name = tc.get("name", "")
            nodeid = f"{classname}::{name}" if classname else name

            if nodeid in seen_nodeids:
                continue
            seen_nodeids.add(nodeid)

            time_val = float(tc.get("time", 0))

            skip_elem = tc.find("skipped")
            failure_elem = tc.find("failure")
            error_elem = tc.find("error")

            if skip_elem is not None:
                skip_type = skip_elem.get("type", "")
                if skip_type == "pytest.xfail":
                    status = "passed"
                    message = "xfailed: expected failure"
                else:
                    status = "skipped"
                    message = skip_elem.get("message", "") or (skip_elem.text or "").strip()
            elif error_elem is not None:
                status = "error"
                message = error_elem.get("message", "") or (error_elem.text or "").strip()
            elif failure_elem is not None:
                status = "failed"
                message = failure_elem.get("message", "") or (failure_elem.text or "").strip()
            else:
                status = "passed"
                message = ""

            # Determine test file from classname
            test_file = classname.split("::")[0] if classname else ""
            if test_file and not test_file.startswith("test/"):
                test_file = "test/" + test_file

            if test_file not in files_cases:
                files_cases[test_file] = []

            files_cases[test_file].append({
                "nodeid": nodeid,
                "status": status,
                "duration": time_val,
                "message": message,
                "case_idx": len(files_cases[test_file]),
            })

    return files_cases


# ==============================================================================
# JSONL Generation
# ==============================================================================


def generate_jsonl(
    category,
    shard,
    expected_files,
    stderr_info,
    files_cases,
    reports_dir,
    runner="linux-aarch64-a3-8",
):
    """Build the JSONL output.

    For each expected file, combine:
      - stderr_info: duration + return_code + message
      - files_cases: per-case results from JUnit XML

    Files with XML but no cases (empty test file) get cases=[].
    Files without XML or stderr info are crash/missing files with cases=[].
    """
    prefix = _PREFIX_MAP.get(category, "reg")
    output_path = Path(reports_dir) / f"shard_{prefix}-{shard}_cases.jsonl"

    per_file_records = []
    total_cases = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_skipped = 0

    for test_file in expected_files:
        test_file = test_file.strip()
        if not test_file:
            continue

        # Normalize: remove .py if present for lookup
        lookup_key = test_file
        if lookup_key.endswith(".py"):
            lookup_key_no_ext = lookup_key[:-3]
        else:
            lookup_key_no_ext = lookup_key

        # Get cases from XML
        cases = files_cases.get(lookup_key_no_ext, files_cases.get(lookup_key, []))

        # Get file-level info from stderr
        fi = stderr_info.get(lookup_key_no_ext, stderr_info.get(lookup_key, None))

        if fi is not None:
            duration = fi["duration"]
            return_code = fi.get("return_code", 0)
            message = fi.get("message", "")
        else:
            # File not mentioned in stderr at all — missing/crashed
            duration = None
            return_code = -1
            message = "No status reported by run_test.py (process may have crashed)"

        # If we have cases, duration from XML is more accurate for passing files
        if cases and duration is None:
            duration = sum(c["duration"] for c in cases)

        # Aggregate case stats
        file_passed = sum(1 for c in cases if c["status"] == "passed")
        file_failed = sum(1 for c in cases if c["status"] == "failed")
        file_errors = sum(1 for c in cases if c["status"] == "error")
        file_skipped = sum(1 for c in cases if c["status"] == "skipped")

        total_cases += len(cases)
        total_passed += file_passed
        total_failed += file_failed
        total_errors += file_errors
        total_skipped += file_skipped

        per_file_records.append({
            "test_file": test_file if test_file.endswith(".py") else test_file + ".py",
            "duration": duration,
            "return_code": return_code if return_code is not None else 1,
            "message": message,
            "cases": cases,
        })

    # Sort by test_file for stable output
    per_file_records.sort(key=lambda r: r["test_file"])

    # Write JSONL
    summary = {
        "shard": shard,
        "shard_type": category,
        "execution_mode": "file_level_upstream",
        "runner": runner,
        "total_files": len(per_file_records),
        "total_cases": total_cases,
        "passed": total_passed,
        "failed": total_failed,
        "errors": total_errors,
        "skipped": total_skipped,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        for rec in per_file_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"{category} shard {shard}: "
          f"{total_passed}P {total_failed}F {total_errors}E {total_skipped}S "
          f"= {total_cases} cases across {len(per_file_records)} files")
    print(f"JSONL saved to {output_path}")

    return output_path


# ==============================================================================
# CLI
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate shard JSONL from JUnit XMLs + run_test.py stderr"
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--expected-files", required=True,
                        help="File listing expected test files (one per line)")
    parser.add_argument("--execution-log", default="",
                        help="run_test.py stderr tee log")
    parser.add_argument("--reports-dir", default="test-reports")
    parser.add_argument("--runner", default="linux-aarch64-a3-8",
                        help="Runner label for this shard")
    args = parser.parse_args()

    # Read expected files
    expected_files = []
    if os.path.isfile(args.expected_files):
        with open(args.expected_files, encoding="utf-8") as f:
            expected_files = [line.strip() for line in f if line.strip()]
    # Populate global set for _match_stderr_to_expected lookups
    _EXPECTED_FILES_SET.update(expected_files)

    # Parse stderr for file-level status
    stderr_info = parse_execution_log(args.execution_log)

    # Parse JUnit XMLs for per-case results
    reports_dir = Path(args.reports_dir)
    xml_files = find_xml_files(reports_dir)
    print(f"Found {len(xml_files)} JUnit XML files")
    files_cases = parse_xml_testcases(xml_files)

    # Generate JSONL
    generate_jsonl(
        args.category,
        args.shard,
        expected_files,
        stderr_info,
        files_cases,
        reports_dir,
        runner=args.runner,
    )


if __name__ == "__main__":
    main()
