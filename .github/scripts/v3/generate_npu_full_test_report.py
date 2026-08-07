#!/usr/bin/env python3
"""
Generate consolidated NPU full test summary report from JSONL shard outputs.

Aggregates per-shard JSONL files into a single markdown report and JSON summary.

JSONL format (one file per shard):
  Line 1 — shard summary:
    {"shard":1,"shard_type":"core","execution_mode":"file_level_upstream",
     "runner":"linux-aarch64-a3-8","total_files":47,"total_cases":1523,
     "passed":1480,"failed":30,"errors":10,"skipped":5}

  Lines 2+ — per-file records:
    {"test_file":"test/nn/test_convolution.py","duration":150.0,
     "return_code":0,"message":"","cases":[{...}]}

Aggregation:
  - totals and by_category: summed from shard summary lines (have "shard" key)
  - by_file: counted from per-file lines (have "test_file" key) via cases[].status
  - crashed_files: identified from per-file lines with return_code < 0

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
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main():
    args = parse_args()
    reports_root = Path(args.reports_root)

    # ── Aggregate from JSONL shard files ──────────────────────────────
    totals = defaultdict(int)
    by_category = defaultdict(lambda: defaultdict(int))
    by_file = {}
    crashed_files = []  # list of (test_file, return_code, message, duration)

    for fpath in sorted(reports_root.rglob("*.jsonl")):
        try:
            with open(fpath, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue

        if not lines:
            continue

        # Line 1: shard summary
        try:
            summary = json.loads(lines[0].strip())
        except json.JSONDecodeError:
            continue

        # Only aggregate from shard summary lines
        if "shard" not in summary:
            continue

        cat = summary.get("shard_type", "unknown")

        totals["passed"] += int(summary.get("passed", 0))
        totals["failed"] += int(summary.get("failed", 0))
        totals["errors"] += int(summary.get("errors", 0))
        totals["skipped"] += int(summary.get("skipped", 0))
        totals["total"] += int(summary.get("total_cases", 0))

        by_category[cat]["passed"] += int(summary.get("passed", 0))
        by_category[cat]["failed"] += int(summary.get("failed", 0))
        by_category[cat]["errors"] += int(summary.get("errors", 0))
        by_category[cat]["skipped"] += int(summary.get("skipped", 0))
        by_category[cat]["total"] += int(summary.get("total_cases", 0))

        # Lines 2+: per-file records
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Must be a per-file record
            fn = rec.get("test_file", "")
            if not fn:
                continue

            return_code = rec.get("return_code", 0)

            # Initialize by_file entry
            if fn not in by_file:
                by_file[fn] = {
                    "passed": 0, "failed": 0, "errors": 0,
                    "skipped": 0, "total": 0,
                    "poisoned": 0,
                    "return_code": return_code,
                    "message": rec.get("message", ""),
                    "duration": rec.get("duration"),
                }

            # Track crashed files (killed by signal)
            if return_code < 0:
                crashed_files.append((
                    fn, return_code, rec.get("message", ""), rec.get("duration"),
                ))

            # Count cases by status from the cases[] array
            for case in rec.get("cases", []):
                status = case.get("status", "unknown")
                by_file[fn]["total"] += 1
                if status == "passed":
                    by_file[fn]["passed"] += 1
                elif status == "failed":
                    by_file[fn]["failed"] += 1
                elif status == "error":
                    by_file[fn]["errors"] += 1
                elif status == "skipped":
                    by_file[fn]["skipped"] += 1
                if case.get("poisoned"):
                    by_file[fn]["poisoned"] += 1

    # ── Generate JSON report ──────────────────────────────────────────
    total_failed = totals["failed"] + totals["errors"]

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
        "crashed_files": [
            {"test_file": fn, "return_code": rc, "message": msg, "duration": dur}
            for fn, rc, msg, dur in crashed_files
        ],
    }

    json_path = Path(args.output_json)
    json_path.write_text(
        json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Generate Markdown report ──────────────────────────────────────
    md_lines = []
    md_lines.append("# NPU Full Test Summary (v3)")
    md_lines.append("")
    md_lines.append(f"**Generated:** {json_report['meta']['generated_at']}")
    md_lines.append(f"**PyTorch version:** {args.pytorch_version}")
    md_lines.append(f"**torch_npu wheel:** {args.torch_npu_whl}")
    md_lines.append(f"**Docker image:** {args.docker_image}")
    md_lines.append(f"**Runner:** {args.runner}")
    md_lines.append("")

    # Overall stats
    md_lines.append("## Overall")
    md_lines.append("")
    md_lines.append("| Metric | Count |")
    md_lines.append("|--------|-------|")
    md_lines.append(f"| Passed | {totals['passed']} |")
    md_lines.append(f"| Failed | {totals['failed']} |")
    md_lines.append(f"| Errors | {totals['errors']} |")
    md_lines.append(f"| Skipped | {totals['skipped']} |")
    md_lines.append(f"| **Total** | **{totals['total']}** |")
    md_lines.append(f"| **Total Failed** | **{total_failed}** |")
    if totals["total"] > 0:
        pass_rate = 100 * totals["passed"] / totals["total"]
        md_lines.append(f"| Pass Rate | {pass_rate:.1f}% |")
    md_lines.append("")

    # By category
    md_lines.append("## By Category")
    md_lines.append("")
    md_lines.append("| Category | Passed | Failed | Errors | Skipped | Total |")
    md_lines.append("|----------|--------|--------|--------|---------|-------|")
    for cat in sorted(by_category.keys()):
        s = by_category[cat]
        md_lines.append(
            f"| {cat} | {s['passed']} | {s['failed']} | {s['errors']} | "
            f"{s['skipped']} | {s['total']} |"
        )
    md_lines.append("")

    # Crashed files (core dumps / signal kills)
    if crashed_files:
        md_lines.append("## Crashed Files (Core Dumps / Signal Kills)")
        md_lines.append("")
        md_lines.append("| Test File | Signal | Return Code | Duration | Message |")
        md_lines.append("|-----------|--------|-------------|----------|---------|")
        for fn, rc, msg, dur in crashed_files:
            signal_name = msg.split(":")[0] if ":" in msg else msg
            dur_str = f"{dur:.1f}s" if dur is not None else "N/A"
            md_lines.append(f"| {fn} | {signal_name} | {rc} | {dur_str} | {msg} |")
        md_lines.append("")

    # Top-level per-file summary (failing/crashed files first)
    md_lines.append("## By File")
    md_lines.append("")
    md_lines.append("| Test File | Passed | Failed | Errors | Skipped | Total | Status |")
    md_lines.append("|-----------|--------|--------|--------|---------|-------|--------|")
    for fn in sorted(by_file.keys()):
        s = by_file[fn]
        tags = []
        if s["return_code"] < 0:
            tags.append(f"CRASHED (rc={s['return_code']})")
        elif s["failed"] > 0 or s["errors"] > 0:
            tags.append("FAILED")
        if s.get("poisoned", 0) > 0:
            tags.append(f"POISONED ({s['poisoned']})")
        if not tags:
            tags = ["OK"] if s["total"] > 0 else ["EMPTY"]
        st = ", ".join(tags)
        md_lines.append(
            f"| {fn} | {s['passed']} | {s['failed']} | {s['errors']} | "
            f"{s['skipped']} | {s['total']} | {st} |"
        )

    md_path = Path(args.output_markdown)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Markdown report: {md_path} ({md_path.stat().st_size} bytes)")
    print(f"JSON report: {json_path} ({json_path.stat().st_size} bytes)")
    print(f"Totals: {totals['passed']} passed, {total_failed} failed")
    if crashed_files:
        print(f"Crashed files: {len(crashed_files)}")
        for fn, rc, msg, dur in crashed_files:
            print(f"  [{rc}] {fn}: {msg}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate consolidated NPU test summary report from JSONL shard outputs"
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
