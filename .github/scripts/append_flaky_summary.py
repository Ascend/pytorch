#!/usr/bin/env python3
"""Merge flaky-case (retry-recovered) summaries into an existing NPU report.

Runs in the custom-test.yml report job right after
generate_npu_full_test_report.py, reading the same downloaded shard JSONs
under --reports-root. A flaky case is one that retry_failed_cases.py marked
retried=true whose final status is passed -- i.e. its first attempt failed /
errored / timed out but a later isolated round recovered it. Cases that were
retried and STILL failed are real failures, not flaky, and stay in the main
report bodies.

    - Non-empty result: appends a "## 不稳定用例 (Flaky Cases)" section to
      --markdown and writes a "flaky_summary" block into --json (merged into
      the existing object, never replacing it), so the main report and the
      step summary both carry the flaky list.
    - Empty result: prints a skip note and changes nothing, so a report never
      grows an empty section.

Always exits 0: the flaky section is supplementary and must never fail the
report job.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

FLAKY_HEADING = "## 不稳定用例 (Flaky Cases)"
# Only a genuine first-attempt failure counts; the retry machinery only ever
# retries these three statuses anyway (mirrors RETRY_STATUSES in
# retry_failed_cases.py).
FLAKY_FIRST_STATUSES = ("failed", "error", "timeout")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append flaky-case (retry-recovered) summary to an NPU full test report"
    )
    parser.add_argument("--reports-root", type=str, required=True,
                        help="Directory holding the downloaded shard_*_cases.json files")
    parser.add_argument("--markdown", type=str, required=True,
                        help="Existing report markdown to append the flaky section to")
    parser.add_argument("--json", type=str, dest="json_path", required=True,
                        help="Existing report JSON to merge the flaky_summary block into")
    return parser.parse_args()


def collect_flaky_cases(reports_root: Path) -> List[Dict]:
    """Return one record per flaky case found across all shard result files."""
    flaky: List[Dict] = []
    for cases_file in sorted(reports_root.rglob("shard_*_cases.json")):
        # Retry evidence lives under a retry/ subdir (attempt_N/...); its
        # interim cases.json files carry no retried marker, but skip them
        # anyway so only the merged main-run results are scanned.
        if "retry" in cases_file.parts:
            continue
        try:
            data = json.loads(cases_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"FLAKY: skipping unreadable {cases_file}: {exc}", file=sys.stderr)
            continue
        shard = data.get("shard")
        for case in data.get("cases", []):
            if case.get("retried") is not True:
                continue
            if case.get("status") != "passed":
                continue
            first = case.get("first_attempt_status")
            if first not in FLAKY_FIRST_STATUSES:
                continue
            flaky.append({
                "nodeid": case.get("nodeid", ""),
                "file": case.get("file", ""),
                "first_attempt_status": first,
                "retry_attempts": case.get("retry_attempts"),
                "shard": shard,
            })
    return flaky


def build_markdown_section(flaky: List[Dict]) -> str:
    """Render the flaky cases as a markdown section."""
    lines = [
        "",
        FLAKY_HEADING,
        "",
        f"- 共 {len(flaky)} 条：首轮失败、重试后通过（不稳定用例），不计入最终失败。",
        "",
        "| # | nodeid | 首轮状态 | 重试轮次 |",
        "|---|--------|----------|----------|",
    ]
    for index, case in enumerate(flaky, start=1):
        attempts = case.get("retry_attempts")
        lines.append(
            f"| {index} | {case['nodeid']} | {case['first_attempt_status']} "
            f"| {attempts if attempts is not None else '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def append_markdown(markdown_path: Path, section: str) -> None:
    with markdown_path.open("a", encoding="utf-8") as handle:
        handle.write(section)


def merge_json(json_path: Path, flaky: List[Dict]) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"report JSON is not an object: {json_path}")
    data["flaky_summary"] = {"count": len(flaky), "cases": flaky}
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    reports_root = Path(args.reports_root)
    markdown_path = Path(args.markdown)
    json_path = Path(args.json_path)

    if not reports_root.is_dir():
        print(f"FLAKY: reports root not found: {reports_root}; nothing to summarise")
        return 0

    flaky = collect_flaky_cases(reports_root)
    if not flaky:
        print("FLAKY: no flaky (retry-recovered) cases; report left unchanged")
        return 0

    print(f"FLAKY: {len(flaky)} flaky case(s) recovered by retry")

    if markdown_path.exists():
        append_markdown(markdown_path, build_markdown_section(flaky))
        print(f"FLAKY: appended {FLAKY_HEADING} to {markdown_path}")
    else:
        print(f"FLAKY: markdown not found, skipping: {markdown_path}", file=sys.stderr)

    if json_path.exists():
        try:
            merge_json(json_path, flaky)
            print(f"FLAKY: merged flaky_summary into {json_path}")
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            print(f"FLAKY: could not update {json_path}: {exc}", file=sys.stderr)
    else:
        print(f"FLAKY: json not found, skipping: {json_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
