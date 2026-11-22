#!/usr/bin/env python3
"""List blacklist nodeids that are candidates for removal.

The restructured blacklist-test pipeline runs the FULL union of blacklist
nodeids on BOTH platforms (a3 and a5). This script reads each platform's shard
reports (reports-a3/ reports-a5/) and collects, per platform, the nodeids that:

  * passed                        -> stale entry (the case now passes);
  * skipped                       -> stale entry (the case is skipped, so it no
                                     longer exercises anything);
  * status == error and message   -> the nodeid matched no collected test
    == "No testcase in XML"          (the test was removed / renamed).

Additionally, for nodeids that are still failing, a log-search verifies whether
the blacklist ``log`` (raw error, possibly ';'-joined) actually appears in the
reported error message. Nodeids whose recorded log does NOT appear are flagged
as ``log_mismatch`` — their blacklist justification is out of sync.

These groups are written to a JSON file and a Markdown report. Nothing is
written back to the repository.

Usage:
    python3 generate_removal_candidates.py \
        --reports-root . \
        --config-dir .github/config/blacklist_testccases \
        --output-prefix removal-candidates
"""

import argparse
import datetime
import importlib.util
import json
import os
import sys
from pathlib import Path

PLATFORMS = ("a3", "a5")
NO_TESTCASE_MSG = "No testcase in XML"
SENTINEL_MESSAGES = {
    "No testcase in XML",
    "XML file not generated",
    "XML parse failed",
}

RULE = (
    "a3 / a5 each run the full union of blacklist nodeids; passed / skipped / "
    '"No testcase in XML" error nodeids are removal candidates; still-failing '
    "nodeids whose recorded log does NOT appear in the actual error are "
    "flagged as log mismatch"
)


def load_results(reports_dir):
    """Return (status_by_nodeid, message_by_nodeid, n_files) for a platform."""
    reports_dir = Path(reports_dir)
    statuses = {}
    messages = {}
    if reports_dir.is_dir():
        paths = sorted(reports_dir.rglob("*_cases.json"))
    else:
        paths = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for case in data.get("cases", []) or []:
            nodeid = case.get("nodeid")
            if not nodeid:
                continue
            statuses[nodeid] = case.get("status", "error")
            messages[nodeid] = case.get("message") or ""
    return statuses, messages, len(paths)


def categorize(statuses, messages):
    """Split nodeids into passed / skipped / no_testcase_in_xml lists."""
    passed = []
    skipped = []
    no_testcase = []
    for nodeid in sorted(statuses):
        status = statuses[nodeid]
        if status == "passed":
            passed.append(nodeid)
        elif status == "skipped":
            skipped.append(nodeid)
        if NO_TESTCASE_MSG in messages.get(nodeid, ""):
            no_testcase.append(nodeid)
    return passed, skipped, no_testcase


def load_flatten_module():
    """Import flatten_blacklist_testccases.py from the repo config scripts dir."""
    script = (
        Path(__file__).resolve().parents[4]
        / ".github/config/blacklist_testccases/scripts/flatten_blacklist_testccases.py"
    )
    spec = importlib.util.spec_from_file_location(
        "flatten_blacklist_testccases", str(script)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_log_mismatch(log_map, statuses, messages, all_logs):
    """Return mismatches whose recorded log does NOT appear in the error.

    Only nodeids that are still failing (status failed/error) with a real error
    message are considered. The ``log`` may hold several ';'-joined fragments;
    the nodeid is flagged when NONE of them appear in the message. Each flagged
    entry carries the recorded ``log`` plus any fragment collected from the
    blacklist groups' ``log`` fields that DOES appear in the message (the
    up-to-date error signature, if any).
    """
    mismatched = []
    for nodeid in sorted(log_map):
        log = (log_map[nodeid] or "").strip()
        if not log:
            continue
        status = statuses.get(nodeid)
        if status not in ("failed", "error"):
            continue
        message = messages.get(nodeid, "") or ""
        if not message or message in SENTINEL_MESSAGES:
            continue
        fragments = [fragment.strip() for fragment in log.split(";") if fragment.strip()]
        if any(fragment in message for fragment in fragments):
            continue
        matched = [fragment for fragment in all_logs if fragment in message]
        mismatched.append(
            {
                "nodeid": nodeid,
                "recorded_log": log,
                "matched_logs": matched,
            }
        )
    return mismatched


def build_report(reports_root, config_dir):
    flatten = load_flatten_module()
    log_map = flatten.collect_log_map(config_dir)
    all_logs = sorted({log for log in log_map.values() if log})

    platforms = {}
    counts = {}
    for platform in PLATFORMS:
        statuses, messages, n_files = load_results(
            Path(reports_root) / f"reports-{platform}"
        )
        passed, skipped, no_testcase = categorize(statuses, messages)
        log_mismatch = build_log_mismatch(log_map, statuses, messages, all_logs)
        platforms[platform] = {
            "passed": passed,
            "skipped": skipped,
            "no_testcase_in_xml": no_testcase,
            "log_mismatch": log_mismatch,
        }
        counts[platform] = {
            "passed": len(passed),
            "skipped": len(skipped),
            "no_testcase_in_xml": len(no_testcase),
            "log_mismatch": len(log_mismatch),
            "report_files": n_files,
            "total_statuses": len(statuses),
        }
    return {
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": RULE,
        "platforms": platforms,
        "counts": counts,
    }


def render_markdown(report):
    lines = ["# 黑名单剔除候选", ""]
    lines.append(
        "> 判定：a3 / a5 各自跑全量 nodeid 并集后，`passed` / `skipped` / "
        "`No testcase in XML`（error）均为可剔除候选；仍在失败的用例中，黑名单 `log` "
        "未出现在实际报错里的记为 `log mismatch`。本报告只列 nodeid，不写回仓库。"
    )
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append("| platform | passed | skipped | no testcase in XML (error) | log mismatch |")
    lines.append("|---|---|---|---|---|")
    for platform in PLATFORMS:
        c = report["counts"][platform]
        lines.append(
            f"| {platform} | {c['passed']} | {c['skipped']} | "
            f"{c['no_testcase_in_xml']} | {c['log_mismatch']} |"
        )
    lines.append("")

    sections = (
        ("passed", "passed"),
        ("skipped", "skipped"),
        ("no_testcase_in_xml", "No testcase in XML (error)"),
        ("log_mismatch", "log mismatch (recorded log not found in error)"),
    )
    for platform in PLATFORMS:
        block = report["platforms"][platform]
        lines.append(f"## {platform}")
        lines.append("")
        for key, label in sections:
            nodeids = block[key]
            lines.append(f"### {label} ({len(nodeids)})")
            lines.append("")
            if key == "log_mismatch":
                lines.append("| nodeid | 黑名单 log | 命中新日志 |")
                lines.append("|---|---|---|")
                for item in nodeids:
                    matched = "; ".join(item["matched_logs"]) or ""
                    lines.append(
                        f"| {item['nodeid']} | {item['recorded_log']} | {matched} |"
                    )
            else:
                lines.append("```")
                lines.extend(nodeids)
                lines.append("```")
            lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-root",
        default=".",
        help="Root containing reports-a3/ and reports-a5/ (default: .)",
    )
    parser.add_argument(
        "--config-dir",
        default=".github/config/blacklist_testccases",
        help="Directory with the grouped blacklist files used for the "
        "nodeid -> log mapping (default: .github/config/blacklist_testccases)",
    )
    parser.add_argument(
        "--output-prefix",
        default="removal-candidates",
        help="Output prefix for the .json / .md files",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_report(args.reports_root, args.config_dir)

    json_path = Path(f"{args.output_prefix}.json")
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    markdown = render_markdown(report)
    md_path = Path(f"{args.output_prefix}.md")
    md_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nwritten: {json_path}, {md_path}")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")


if __name__ == "__main__":
    main()