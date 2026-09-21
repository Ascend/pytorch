#!/usr/bin/env python3
"""
Summarize cases_collection_summary.json files produced by collect jobs.

Renders a per-category case-count comparison table across one or more
collections to stdout and $GITHUB_STEP_SUMMARY (when available).

Usage:
    python3 summarize_collection.py \
        --label "A3 (aarch64)" --summary a3/cases_collection_summary.json \
        --label "A5 (x86_64)" --summary a5/cases_collection_summary.json

Exit code is always 0; missing or unreadable summary files render as a
"(missing)" column instead of failing the job.
"""

import argparse
import json
import os
import sys

CATEGORY_ORDER = ["core", "tensor", "distributed", "graph", "others"]


def load_summary(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARNING: cannot read {path}: {e}", file=sys.stderr)
        return None


def render(labels, summaries):
    valid = [(l, s) for l, s in zip(labels, summaries) if s is not None]
    show_delta = len(valid) == 2

    cats = [c for c in CATEGORY_ORDER if any(c in s.get("categories", {}) for _, s in valid)]
    for _, s in valid:
        for c in s.get("categories", {}):
            if c not in cats:
                cats.append(c)

    def cell(v, bold=False):
        if v is None:
            return "-"
        return f"**{v}**" if bold else str(v)

    header = ["Category"] + [
        l if s is not None else f"{l} (missing)" for l, s in zip(labels, summaries)
    ]
    if show_delta:
        header.append("Delta")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    def delta(a, b, bold=False):
        if a is None or b is None:
            return "-"
        d = b - a
        text = f"{d:+d}" if d else "0"
        return f"**{text}**" if bold else text

    for cat in cats:
        vals = [s.get("categories", {}).get(cat, {}).get("total_cases") if s else None
                for s in summaries]
        row = [cat] + [cell(v) for v in vals]
        if show_delta:
            row.append(delta(vals[0], vals[1]))
        lines.append("| " + " | ".join(row) + " |")

    totals = [s.get("total_cases") if s else None for s in summaries]
    row = ["**Total cases**"] + [cell(t, bold=True) for t in totals]
    if show_delta:
        row.append(delta(totals[0], totals[1], bold=True))
    lines.append("| " + " | ".join(row) + " |")

    for key, name in (("total_files", "Total files"), ("total_skipped", "Total skipped")):
        vals = [s.get(key) if s else None for s in summaries]
        row = [name] + [cell(v) for v in vals]
        if show_delta:
            row.append(delta(vals[0], vals[1]))
        lines.append("| " + " | ".join(row) + " |")

    if show_delta:
        lines.append("")
        lines.append(f"*Delta = {labels[1]} − {labels[0]}*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize cases_collection_summary.json files side by side."
    )
    parser.add_argument("--label", action="append", required=True,
                        help="Display label for a collection (repeatable)")
    parser.add_argument("--summary", action="append", required=True,
                        help="Path to cases_collection_summary.json (repeatable, pairs with --label)")
    args = parser.parse_args()
    if len(args.label) != len(args.summary):
        parser.error("--label and --summary must appear in pairs")

    summaries = [load_summary(p) for p in args.summary]
    table = render(args.label, summaries)
    print(table)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write("## Case Collection Summary\n\n")
            f.write(table + "\n")


if __name__ == "__main__":
    main()
