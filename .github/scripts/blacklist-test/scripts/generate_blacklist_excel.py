#!/usr/bin/env python3
"""Generate the blacklist reports: a 3-sheet Excel plus a flat per-nodeid JSON.

The blacklist-test pipeline now exercises the full blacklist universe U on each
platform: first the blacklist run (U minus the nodeids recorded as passed), then
a pass run (the nodeids recorded as passed on that platform). This script joins
those four result sets into one workbook:

  Sheet 1 "公共黑名单" : nodeids still blacklisted on BOTH platforms,
                         i.e. U - passed_a3 - passed_a5. Carries the blacklist
                         run status on a3 and a5.
  Sheet 2 "a3 pass"   : nodeids recorded as passed on a3, with their a3 pass
                        run status.
  Sheet 3 "a5 pass"   : nodeids recorded as passed on a5, with their a5 pass
                        run status.

Every row also carries the nodeid's blacklist metadata (category / reason /
issue) looked up from the blacklist config files.

The companion JSON is a flat array covering the whole union U, one object per
nodeid:

    {"nodeid": ..., "a3_status": ..., "a5_status": ...,
     "a3_log": ..., "a5_log": ...}

Statuses/logs come from the blacklist run when the nodeid was executed there,
otherwise from the platform's pass run. Unknown nodeids get null.

Usage:
    python3 generate_blacklist_excel.py \
        --reports-root . \
        --config-dir .github/config/blacklist_testccases \
        --output blacklist-report.xlsx \
        --json-output blacklist-report.json
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

PLATFORMS = ("a3", "a5")
MISSING = "未执行"
HEADERS_COMMON = ["nodeid", "category", "reason", "issue", "a3结果", "a5结果"]
HEADERS_PASS = ["nodeid", "category", "reason", "issue"]


def load_flatten_module(config_dir):
    script = Path(config_dir) / "scripts" / "flatten_blacklist_testccases.py"
    spec = importlib.util.spec_from_file_location("flatten_blacklist_testccases", str(script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_meta_map(config_dir):
    """Return {nodeid: {category, reason, issue}} across the blacklist files.

    Handles both file shapes used under blacklist_testccases/:
      * ``*_unsupported_testcases.json``: list of {log: [..], category, nodeids: {..}}
      * ``*_grouped.json``: {groups: [{category, reason, issue, nodes: [..]}]}
    """
    meta = {}
    for path in sorted(Path(config_dir).glob("**/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                category = entry.get("category") or ""
                log = entry.get("log")
                if isinstance(log, list):
                    reason = "; ".join(item for item in log if isinstance(item, str) and item)
                else:
                    reason = log or ""
                nodeids = entry.get("nodeids")
                if isinstance(nodeids, dict):
                    for nodeid, val in nodeids.items():
                        issue = ""
                        if isinstance(val, dict):
                            issue = val.get("issue") or ""
                        meta[nodeid] = {
                            "category": category,
                            "reason": reason,
                            "issue": issue,
                        }
        elif isinstance(data, dict):
            for group in data.get("groups", []) or []:
                if not isinstance(group, dict):
                    continue
                category = group.get("category") or ""
                reason = group.get("reason") or ""
                issue = group.get("issue") or ""
                for node in group.get("nodes") or []:
                    if isinstance(node, str) and node:
                        meta[node] = {
                            "category": category,
                            "reason": reason,
                            "issue": issue,
                        }
    return meta


def load_results(reports_root, name):
    """Return {nodeid: {"status": str, "message": str}} merged from *_cases.json."""
    reports_dir = Path(reports_root) / name
    results = {}
    if not reports_dir.is_dir():
        return results
    for path in sorted(reports_dir.rglob("*_cases.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for case in data.get("cases", []) or []:
            nodeid = case.get("nodeid")
            if nodeid:
                results[nodeid] = {
                    "status": case.get("status", "error"),
                    "message": case.get("message") or "",
                }
    return results


def status_of(results, nodeid):
    """Return the status string for `nodeid`, or the MISSING placeholder."""
    entry = results.get(nodeid)
    return entry["status"] if entry else MISSING


def build_report(reports_root, config_dir):
    flatten = load_flatten_module(config_dir)
    union, _stats = flatten.collect_nodeids(config_dir)
    union = set(union)

    diff = json.loads(
        (Path(config_dir) / "conf" / "platform_passed_diff.json").read_text(encoding="utf-8")
    )
    passed = {
        platform: [n for n in (diff.get(f"passed_{platform}", []) or []) if n]
        for platform in PLATFORMS
    }

    common = sorted(union - set(passed["a3"]) - set(passed["a5"]))
    meta = build_meta_map(config_dir)

    blacklist_status = {
        platform: load_results(reports_root, f"reports-{platform}")
        for platform in PLATFORMS
    }
    pass_status = {
        platform: load_results(reports_root, f"reports-pa{platform[1]}")
        for platform in PLATFORMS
    }

    def meta_row(nodeid):
        m = meta.get(nodeid) or {}
        return [
            nodeid,
            m.get("category", ""),
            m.get("reason", ""),
            m.get("issue", ""),
        ]

    common_rows = [
        meta_row(n)
        + [
            status_of(blacklist_status["a3"], n),
            status_of(blacklist_status["a5"], n),
        ]
        for n in common
    ]
    a3_rows = [meta_row(n) + [status_of(pass_status["a3"], n)] for n in sorted(passed["a3"])]
    a5_rows = [meta_row(n) + [status_of(pass_status["a5"], n)] for n in sorted(passed["a5"])]

    return {
        "common": common_rows,
        "a3": a3_rows,
        "a5": a5_rows,
        "counts": {
            "common": len(common_rows),
            "a3": len(a3_rows),
            "a5": len(a5_rows),
            "union": len(union),
        },
    }


def build_json_rows(reports_root, config_dir):
    """Return flat per-nodeid rows covering the whole blacklist union.

    For each nodeid the blacklist-run entry wins over that platform's pass-run
    entry; a nodeid present in neither gets null status/log.
    """
    flatten = load_flatten_module(config_dir)
    union, _stats = flatten.collect_nodeids(config_dir)

    details = {}
    for platform in PLATFORMS:
        # Pass-run entries first so a blacklist-run entry (if any) wins.
        merged = load_results(reports_root, f"reports-pa{platform[1]}")
        merged.update(load_results(reports_root, f"reports-{platform}"))
        details[platform] = merged

    rows = []
    for nodeid in union:
        a3 = details["a3"].get(nodeid)
        a5 = details["a5"].get(nodeid)
        rows.append(
            {
                "nodeid": nodeid,
                "a3_status": a3["status"] if a3 else None,
                "a5_status": a5["status"] if a5 else None,
                "a3_log": a3["message"] if a3 else None,
                "a5_log": a5["message"] if a5 else None,
            }
        )
    return rows


def write_json_report(rows, output):
    Path(output).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_workbook(report, output):
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    sheets = [
        ("公共黑名单", HEADERS_COMMON, report["common"]),
        ("a3 pass", HEADERS_PASS + ["a3结果"], report["a3"]),
        ("a5 pass", HEADERS_PASS + ["a5结果"], report["a5"]),
    ]
    for idx, (title, headers, rows) in enumerate(sheets):
        ws = wb.active if idx == 0 else wb.create_sheet()
        ws.title = title
        ws.append(headers)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for row in rows:
            ws.append(row)
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
    wb.save(output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-root",
        default=".",
        help="Root containing reports-a3/reports-a5/reports-pa3/reports-pa5 (default: .)",
    )
    parser.add_argument(
        "--config-dir",
        default=".github/config/blacklist_testccases",
        help="Directory with the blacklist config files (default: "
        ".github/config/blacklist_testccases)",
    )
    parser.add_argument(
        "--output",
        default="blacklist-report.xlsx",
        help="Output .xlsx path (default: blacklist-report.xlsx)",
    )
    parser.add_argument(
        "--json-output",
        default="blacklist-report.json",
        help="Output JSON path with nodeid/a3_status/a5_status/a3_log/a5_log "
        "(default: blacklist-report.json)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_report(args.reports_root, args.config_dir)
    write_workbook(report, args.output)

    json_rows = build_json_rows(args.reports_root, args.config_dir)
    write_json_report(json_rows, args.json_output)

    counts = report["counts"]
    lines = [
        "### Blacklist Excel report",
        "",
        f"- union U: {counts['union']}",
        f"- 公共黑名单 (U - passed_a3 - passed_a5): {counts['common']}",
        f"- a3 pass: {counts['a3']}",
        f"- a5 pass: {counts['a5']}",
        f"- JSON rows: {len(json_rows)}",
        "",
    ]
    body = "\n".join(lines)
    print(body)
    print(f"written: {args.output}")
    print(f"written: {args.json_output}")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(body + "\n")


if __name__ == "__main__":
    main()