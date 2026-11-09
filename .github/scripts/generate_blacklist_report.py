#!/usr/bin/env python3
"""
Generate a blacklist-only report aggregated by reason.

The blacklist-test pipeline executes nodeids taken directly from the
platform blacklist files (no collection phase) and records per-shard case
results in ``shard_*_cases.json``.  This script joins those results back to
the blacklist entries and prints a Markdown report that compares a3 and a5
side by side, grouped by the entry's ``reason``.

Usage:
    python3 generate_blacklist_report.py \
        --reports-root . \
        --platform all \
        --output blacklist-report.md

The report is written to ``--output`` (if given), echoed to stdout, and
appended to ``$GITHUB_STEP_SUMMARY`` when that variable is set.
"""

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Must stay in sync with _blacklist_shard.yml (order matters: first wins).
BLACKLIST_FILES = [
    "disabled_testcases_{platform}_all.json",
    "unsupported_aclnn_testcases.json",
    "unsupport_dtype_testcases.json",
]

STATUS_KEYS = ("passed", "failed", "error", "timeout", "skipped")
NO_REASON = "(no reason)"
NO_RESULT = "missing"


def load_collect_module(scripts_dir: Path):
    """Import collect_all_cases.py so we reuse its skip-list parser."""
    path = scripts_dir / "collect_all_cases.py"
    spec = importlib.util.spec_from_file_location("collect_all_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_blacklist(module, config_dir: Path, platform: str):
    """Return {nodeid: metadata} for one platform (missing files are ignored)."""
    files = [str(config_dir / name.format(platform=platform)) for name in BLACKLIST_FILES]
    return module.load_skip_list(files, platform=platform) or {}


def load_case_status(reports_dir: Path):
    """Return {nodeid: status} merged from all shard_*_cases.json files."""
    status_by_nodeid = {}
    if not reports_dir.is_dir():
        return status_by_nodeid

    for path in sorted(reports_dir.rglob("shard_*_cases.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: cannot read {path}: {exc}", file=sys.stderr)
            continue
        for case in data.get("cases", []) or []:
            nodeid = case.get("nodeid")
            if nodeid:
                status_by_nodeid[nodeid] = case.get("status", "error")
    return status_by_nodeid


def aggregate(blacklist, status_by_nodeid):
    """Aggregate counts per reason for one platform."""
    totals = Counter()
    by_reason = defaultdict(Counter)

    for nodeid, meta in blacklist.items():
        meta = meta or {}
        reason = str(meta.get("reason") or NO_REASON)
        status = status_by_nodeid.get(nodeid, NO_RESULT)
        if status not in STATUS_KEYS:
            status = "error" if status != NO_RESULT else NO_RESULT

        by_reason[reason]["blacklist"] += 1
        by_reason[reason][status] += 1
        totals["blacklist"] += 1
        totals[status] += 1

    return totals, by_reason


def _missing(counter):
    """Blacklist entries with no execution result."""
    return counter["blacklist"] - sum(counter[s] for s in STATUS_KEYS)


def _escape(text):
    return str(text).replace("|", "\\|")


def _num(value, bold=False):
    text = f"{value:,}"
    return f"**{text}**" if bold and value else text


def _signed(value):
    return f"{value:+,}" if value else "0"


def render_overview(platform_totals):
    """One row per platform (a5 below a3) plus a signed diff row."""
    platforms = list(platform_totals)
    metrics = [
        ("黑名单条数", lambda t: t["blacklist"], False),
        ("passed", lambda t: t["passed"], True),
        ("failed", lambda t: t["failed"], False),
        ("error", lambda t: t["error"], False),
        ("timeout", lambda t: t["timeout"], False),
        ("skipped", lambda t: t["skipped"], False),
        ("无结果", _missing, False),
    ]

    header = ["platform"] + [label for label, _, _ in metrics]
    # First column (text) stays left-aligned, numeric columns are centred.
    lines = ["| " + " | ".join(header) + " |", "|---|" + ":---:|" * (len(header) - 1)]

    values = {}
    for platform in platforms:
        values[platform] = [getter(platform_totals[platform]) for _, getter, _ in metrics]
        cells = [platform] + [
            _num(v, bold) for v, (_, _, bold) in zip(values[platform], metrics)
        ]
        lines.append("| " + " | ".join(cells) + " |")

    if len(platforms) == 2:
        first, second = values[platforms[0]], values[platforms[1]]
        cells = [f"差值 ({platforms[1]}−{platforms[0]})"]
        cells.extend(_signed(b - a) for a, b in zip(first, second))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def render_compare(key_header, per_platform, top=None):
    """Compare platforms side by side per reason.

    Plain Markdown only (no HTML): a single header row; the a5/a3 order is
    stated once above the table, so no header cell needs merging or <br>.
    """
    platforms = list(per_platform)
    paired = len(platforms) == 2
    status_groups = ("passed", "failed", "error", "timeout", "skipped", NO_RESULT)
    group_labels = {NO_RESULT: "无结果"}

    def total_of(key):
        return sum(per_platform[p].get(key, Counter())["blacklist"] for p in platforms)

    keys = {key for p in platforms for key in per_platform[p]}
    ordered = sorted(keys, key=lambda key: (-total_of(key), key))
    shown = ordered[:top] if top else ordered
    rest = ordered[len(shown):]

    def value(counter, key):
        if counter is None:
            return None
        if key == "blacklist":
            return counter["blacklist"]
        if key == NO_RESULT:
            return _missing(counter)
        return counter[key]

    def fmt(number, key=None):
        if number is None:
            return "—"
        text = f"{number:,}"
        return f"**{text}**" if key == "passed" and number else text

    rows = [(key, [per_platform[p].get(key) for p in platforms]) for key in shown]
    if rest:
        merged = {p: Counter() for p in platforms}
        for key in rest:
            for p in platforms:
                merged[p] += per_platform[p].get(key, Counter())
        rows.append((None, [merged[p] if merged[p] else None for p in platforms]))

    if paired:
        header = [key_header, "总数"] + [group_labels.get(g, g) for g in status_groups]
    else:
        header = [key_header, "条数"] + [group_labels.get(g, g) for g in status_groups]

    # Pad the reason column with plain spaces so the raw Markdown lines up.
    # Markdown has no syntax for column width: rendered widths are chosen by
    # the viewer, and space/&nbsp; padding does not widen a rendered column.
    labels = [
        f"其余 {len(rest)} 种（合计）" if key is None else _escape(key)
        for key, _ in rows
    ]
    reason_width = max([len(str(header[0]))] + [len(lbl) for lbl in labels] + [40])
    header[0] = str(header[0]).ljust(reason_width)

    lines = []
    if paired:
        # Plain Markdown cannot merge header cells vertically, so the platform
        # order is stated once here instead of repeating "a5/a3" in every header.
        lines.extend([f"> 除 `{key_header}` 外，所有数值列均为「a5 / a3」（左 a5、右 a3）。", ""])
    lines.extend(
        [
            "| " + " | ".join(header) + " |",
            "|---|" + ":---:|" * (len(header) - 1),
        ]
    )

    def pair(c0, c1, group):
        v0, v1 = value(c0, group), value(c1, group)
        if v0 is None and v1 is None:
            return "—"
        text = f"{fmt(v1)} / {fmt(v0)}"
        return f"**{text}**" if group == "passed" and (v0 or v1) else text

    for (key, counters), label in zip(rows, labels):
        label = label.ljust(reason_width)
        if paired:
            c0, c1 = counters
            cells = [label, pair(c0, c1, "blacklist")]
            cells.extend(pair(c0, c1, group) for group in status_groups)
        else:
            c0 = counters[0]
            cells = [label, fmt(value(c0, "blacklist"))]
            cells.extend(fmt(value(c0, group), group) for group in status_groups)
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("> `—` 表示该条目不存在于对应平台黑名单。")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a blacklist report grouped by reason and issue."
    )
    parser.add_argument(
        "--reports-root",
        default=".",
        help="Root directory containing reports-<platform>/ (default: .)",
    )
    parser.add_argument(
        "--platform",
        default="all",
        choices=("a3", "a5", "all"),
        help="Which platform to report (default: all)",
    )
    parser.add_argument(
        "--config-dir",
        default=".github/config",
        help="Directory holding the blacklist JSON files (default: .github/config)",
    )
    parser.add_argument(
        "--scripts-dir",
        default=".github/scripts",
        help="Directory holding collect_all_cases.py (default: .github/scripts)",
    )
    parser.add_argument(
        "--output",
        default="blacklist-report.md",
        help="Markdown output path (default: blacklist-report.md)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Rows per comparison table before merging the rest (default: 25, 0 = all)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    reports_root = Path(args.reports_root)
    config_dir = Path(args.config_dir)
    module = load_collect_module(Path(args.scripts_dir))

    platforms = ["a3", "a5"] if args.platform == "all" else [args.platform]
    platform_totals = {}
    reasons = {}

    for platform in platforms:
        blacklist = load_blacklist(module, config_dir, platform)
        reports_dir = reports_root / f"reports-{platform}"
        if not reports_dir.is_dir():
            reports_dir = reports_root
        status_by_nodeid = load_case_status(reports_dir)
        totals, by_reason = aggregate(blacklist, status_by_nodeid)
        platform_totals[platform] = totals
        reasons[platform] = by_reason

    sections = [
        "# Blacklist Report",
        "",
        "黑名单执行结果，按 reason 对比各平台。"
        "`无结果` = 黑名单条目在本次执行中没有对应用例结果。",
        "",
        "## 总览",
        "",
        render_overview(platform_totals),
        "",
        f"## 按 reason 汇总（Top {args.top}）" if args.top else "## 按 reason 汇总",
        "",
        render_compare("reason", reasons, top=args.top or None),
        "",
    ]
    report = "\n".join(sections)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Blacklist report written to {args.output}")

    print(report)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
