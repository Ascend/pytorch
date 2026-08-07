#!/usr/bin/env python3
"""
Generate consolidated NPU full test summary report.

Aggregates per-shard case result JSONs from all categories into a
single markdown report and JSON summary.

Usage:
    python generate_npu_full_test_report.py \
        --reports-root all-test-reports \
        --output-markdown npu-full-test-summary.md \
        --output-json npu-full-test-summary.json \
        --pytorch-version "nightly" \
        --torch-npu-whl "source-build" \
        --shard-matrix-json '["core-1","dist-1"]'
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main():
    args = parse_args()
    reports_root = Path(args.reports_root)

    all_results = []

    # Walk all JSON/JSONL files under reports root
    for fpath in sorted(reports_root.rglob("*.json")):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if "shard" in data and "test_type" in data:
            all_results.append(data)

    for fpath in sorted(reports_root.rglob("*.jsonl")):
        try:
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "_meta" in obj:
                        continue
                    all_results.append(obj)
        except (json.JSONDecodeError, OSError):
            continue

    # Aggregate totals
    totals = defaultdict(int)
    by_category = defaultdict(lambda: defaultdict(int))
    by_file = {}

    for r in all_results:
        cat = r.get("test_type", r.get("category", "unknown"))
        passed = int(r.get("passed", 0))
        failed = int(r.get("failed", 0))
        errors = int(r.get("errors", 0))
        skipped = int(r.get("skipped", 0))
        timeout = int(r.get("timeout", 0))
        total = int(r.get("total_cases", 0))

        totals["passed"] += passed
        totals["failed"] += failed
        totals["errors"] += errors
        totals["skipped"] += skipped
        totals["timeout"] += timeout
        totals["total"] += total

        by_category[cat]["passed"] += passed
        by_category[cat]["failed"] += failed
        by_category[cat]["errors"] += errors
        by_category[cat]["skipped"] += skipped
        by_category[cat]["timeout"] += timeout
        by_category[cat]["total"] += total

        # Per-file results from nested "files" list
        for file_entry in r.get("files", []):
            fn = file_entry.get("file", file_entry.get("test_file", ""))
            if not fn:
                continue
            if fn not in by_file:
                by_file[fn] = {"passed": 0, "failed": 0, "errors": 0,
                               "skipped": 0, "timeout": 0, "total": 0}
            by_file[fn]["passed"] += int(file_entry.get("passed", 0))
            by_file[fn]["failed"] += int(file_entry.get("failed", 0))
            by_file[fn]["errors"] += int(file_entry.get("errors", 0))
            by_file[fn]["skipped"] += int(file_entry.get("skipped", 0))
            by_file[fn]["timeout"] += int(file_entry.get("timeout", 0))
            # per-file entry in shard_result has no "total_cases" key —
            # compute total from the four status counters.
            by_file[fn]["total"] += (
                int(file_entry.get("passed", 0))
                + int(file_entry.get("failed", 0))
                + int(file_entry.get("errors", 0))
                + int(file_entry.get("skipped", 0))
            )

    # Generate JSON report
    total_failed = totals["failed"] + totals["errors"] + totals["timeout"]
    json_report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pytorch_version": args.pytorch_version,
            "torch_npu_whl": args.torch_npu_whl,
            "docker_image": args.docker_image,
            "runner": args.runner,
        },
        "totals": dict(totals),
        "total_failed": total_failed,
        "by_category": {
            cat: dict(stats) for cat, stats in sorted(by_category.items())
        },
        "by_file": by_file,
    }

    json_path = Path(args.output_json)
    json_path.write_text(
        json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Generate Markdown report
    md_lines = []
    md_lines.append("# NPU Full Test Summary (v3)")
    md_lines.append("")
    md_lines.append(f"**Generated:** {json_report['meta']['generated_at']}")
    md_lines.append(f"**PyTorch version:** {args.pytorch_version}")
    md_lines.append(f"**torch_npu wheel:** {args.torch_npu_whl}")
    md_lines.append(f"**Docker image:** {args.docker_image}")
    md_lines.append(f"**Runner:** {args.runner}")
    md_lines.append("")

    md_lines.append("## Overall")
    md_lines.append("")
    md_lines.append("| Metric | Count |")
    md_lines.append("|--------|-------|")
    md_lines.append(f"| Passed | {totals['passed']} |")
    md_lines.append(f"| Failed | {totals['failed']} |")
    md_lines.append(f"| Errors | {totals['errors']} |")
    md_lines.append(f"| Timeout | {totals['timeout']} |")
    md_lines.append(f"| Skipped | {totals['skipped']} |")
    md_lines.append(f"| **Total** | **{totals['total']}** |")
    md_lines.append(f"| **Total Failed** | **{total_failed}** |")
    if totals["total"] > 0:
        pass_rate = 100 * totals["passed"] / totals["total"]
        md_lines.append(f"| Pass Rate | {pass_rate:.1f}% |")
    md_lines.append("")

    md_lines.append("## By Category")
    md_lines.append("")
    md_lines.append("| Category | Passed | Failed | Errors | Timeout | Skipped | Total |")
    md_lines.append("|----------|--------|--------|--------|---------|---------|-------|")
    for cat in sorted(by_category.keys()):
        s = by_category[cat]
        md_lines.append(
            f"| {cat} | {s['passed']} | {s['failed']} | {s['errors']} | "
            f"{s['timeout']} | {s['skipped']} | {s['total']} |"
        )
    md_lines.append("")

    md_path = Path(args.output_markdown)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Markdown report: {md_path} ({md_path.stat().st_size} bytes)")
    print(f"JSON report: {json_path} ({json_path.stat().st_size} bytes)")
    print(f"Totals: {totals['passed']} passed, {total_failed} failed")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate consolidated NPU test summary report"
    )
    parser.add_argument("--reports-root", required=True,
                        help="Root directory with downloaded test-reports artifacts")
    parser.add_argument("--output-markdown", default="npu-full-test-summary.md",
                        help="Output markdown file")
    parser.add_argument("--output-json", default="npu-full-test-summary.json",
                        help="Output JSON file")
    parser.add_argument("--pytorch-version", default="nightly")
    parser.add_argument("--torch-npu-whl", default="source-build")
    parser.add_argument("--patch-count", default="0")
    parser.add_argument("--shard-matrix-json", default="[]")
    parser.add_argument("--docker-image", default="")
    parser.add_argument("--runner", default="")
    parser.add_argument("--cases-summary", default="")
    parser.add_argument("--cases-by-file-dir", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
