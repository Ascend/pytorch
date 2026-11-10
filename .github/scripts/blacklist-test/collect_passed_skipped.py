#!/usr/bin/env python3
"""Collect blacklist cases that PASSED or were SKIPPED and re-run them on the
other platform.

The blacklists are edited over time; an entry that now passes (or is skipped)
on its own platform is stale there.  These same cases are the natural
candidates to *also* be blacklisted on the OTHER platform, but that can only be
decided by actually running them there.  This script:

  collect  reads the per-platform blacklist reports (reports-a3/ reports-a5/),
           intersects them with each platform's effective blacklist, keeps the
           entries whose status is ``passed`` or ``skipped`` (stale), and emits
           one aggregate file describing them plus retest shard JSONs for the
           direction "run on the other platform".  Entries already listed in
           the target platform's blacklist are NOT retested (their target status
           is already known from that platform's own blacklist report).

  report   reads the aggregate file and the retest reports
           (retest-reports-a3/ retest-reports-a5/), and writes a Markdown
           report of the cross-platform outcome.

Nothing is written back to the repository: the outputs are artifacts only.

Usage:
    python collect_passed_skipped.py collect \
        --reports-root . --config-dir .github/config \
        --output-prefix passed-skipped
    python collect_passed_skipped.py report \
        --input passed-skipped.json --reports-root . \
        --output passed-skipped-retest.md
"""

import argparse
import collections
import datetime
import importlib.util
import json
import os
import sys
from pathlib import Path

PLATFORMS = ("a3", "a5")
OTHER = {"a3": "a5", "a5": "a3"}

# Effective per-platform blacklist file list (same files _blacklist_shard.yml
# shards from).
SKIP_LIST_FILES = [
    "disabled_testcases_{platform}_all.json",
    "unsupported_aclnn_testcases.json",
    "unsupport_dtype_testcases.json",
]

STALE_STATUSES = ("passed", "skipped")
REPORT_STATUSES = ("passed", "failed", "skipped", "error", "timeout")

# Number of retest shards: keep it simple and deterministic.
SHARD_CASE_THRESHOLD = 64

RULE = (
    "cases that PASSED or were SKIPPED on one platform are re-run on the other "
    "platform; cases already listed in the target platform's blacklist are "
    "excluded"
)

NOT_RETESTED = "not-retested"
MISSING = "missing"


# ==============================================================================
# Loading helpers
# ==============================================================================


def load_collect_module():
    """Import collect_all_cases.py from the parent .github/scripts (via importlib)."""
    script = Path(__file__).resolve().parent.parent / "collect_all_cases.py"
    spec = importlib.util.spec_from_file_location("collect_all_cases", str(script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_blacklist(module, config_dir, platform):
    """Load one platform's effective blacklist {nodeid: meta}."""
    files = [str(Path(config_dir) / tpl.format(platform=platform))
             for tpl in SKIP_LIST_FILES]
    return module.load_skip_list(files, platform=platform)


def load_statuses(reports_dir):
    """Build {nodeid: status} from every *_cases.json under reports_dir."""
    statuses = {}
    reports_dir = Path(reports_dir)
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
            if nodeid:
                statuses[nodeid] = case.get("status", "error")
    return statuses, len(paths)


# ==============================================================================
# collect
# ==============================================================================


def build_stale(platform, statuses, blacklist, other_blacklist, other_statuses):
    """Return (records, status_counts, retest_records) for one platform."""
    records = []
    target = OTHER[platform]
    for nodeid in sorted(blacklist):
        status = statuses.get(nodeid)
        if status not in STALE_STATUSES:
            continue
        meta = blacklist[nodeid] or {}
        listed = nodeid in other_blacklist
        records.append({
            "nodeid": nodeid,
            "origin_platform": platform,
            "origin_status": status,
            "source_file": meta.get("source"),
            "source_group": meta.get("platform"),
            "meta": {k: v for k, v in meta.items() if k not in ("source", "platform")},
            "target_platform": target,
            "target_already_listed": listed,
            "target_status_if_known": other_statuses.get(nodeid) if listed else None,
        })
    status_counts = dict(collections.Counter(r["origin_status"] for r in records))
    retest = [r for r in records if not r["target_already_listed"]]
    return records, status_counts, retest


def split_records(records, num_shards):
    """Split records evenly into num_shards buckets."""
    base = len(records) // num_shards
    remainder = len(records) % num_shards
    shards = []
    start = 0
    for i in range(num_shards):
        size = base + (1 if i < remainder else 0)
        shards.append(records[start:start + size])
        start += size
    return shards


def write_retest_shards(target, records):
    """Write retest shard JSONs for cases to run ON ``target``; return summary."""
    num_shards = 1 if len(records) <= SHARD_CASE_THRESHOLD else 2
    out_dir = Path(f"retest-cases-{target}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, shard_records in enumerate(split_records(records, num_shards), 1):
        cases = [
            {"nodeid": r["nodeid"], "file": r["nodeid"].split("::", 1)[0]}
            for r in shard_records
        ]
        data = {
            "shard": i,
            "num_shards": num_shards,
            "test_type": "regular",
            "total_cases": len(cases),
            "cases": cases,
        }
        path = out_dir / f"retest-{target}_cases_shard_{i}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [{target}] retest shard {i}/{num_shards}: {len(cases)} cases -> {path}")
    return {"num_shards": num_shards, "count": len(records)}


def _md_cell(value):
    """Render a value as a Markdown table cell (pipes escaped, None -> '')."""
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def render_collect_markdown(aggregate):
    """Render the human record of the collect phase."""
    lines = ["## Blacklist stale cases (passed / skipped)", ""]
    lines.append(f"- generated_at: {aggregate['generated_at']}")
    lines.append(f"- rule: {aggregate['rule']}")
    lines.append("")

    for platform in PLATFORMS:
        block = aggregate["platforms"][platform]
        cases = block["cases"]
        listed = sum(1 for c in cases if c["target_already_listed"])
        status_counts = block["status_counts"]
        # The retest block is keyed by the platform the retest runs ON, i.e.
        # the other platform's stale set is the one re-run here.
        retest = aggregate["retest"][OTHER[platform]]
        lines.append(
            f"### {platform} (stale on {platform}, retested on {OTHER[platform]})"
        )
        lines.append("")
        lines.append(
            f"- blacklist size: {block['blacklist_total']} | "
            f"passed: {status_counts.get('passed', 0)} | "
            f"skipped: {status_counts.get('skipped', 0)} | "
            f"stale total: {block['stale_total']} | "
            f"already on {OTHER[platform]} blacklist: {listed} | "
            f"retest on {OTHER[platform]}: {retest['count']} "
            f"({retest['num_shards']} shard(s))"
        )
        lines.append("")
        lines.append(
            "| # | nodeid | origin | origin status | source file | source group | "
            "already on target blacklist | known target status |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for idx, case in enumerate(cases, 1):
            lines.append(
                f"| {idx} | `{_md_cell(case['nodeid'])}` | {case['origin_platform']} | "
                f"{case['origin_status']} | {_md_cell(case['source_file'])} | "
                f"{_md_cell(case['source_group'])} | "
                f"{'yes' if case['target_already_listed'] else 'no'} | "
                f"{_md_cell(case['target_status_if_known'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_github_output(outputs):
    """Append key=value lines to $GITHUB_OUTPUT when it is set."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            for key, value in outputs.items():
                handle.write(f"{key}={value}\n")
    except OSError:
        pass


def run_collect(args):
    reports_root = Path(args.reports_root)
    config_dir = Path(args.config_dir)

    try:
        module = load_collect_module()
    except Exception as e:  # noqa: BLE001 - never raise
        print(f"WARNING: could not import collect_all_cases.py: {e}", file=sys.stderr)
        module = None

    loaded = {}
    for platform in PLATFORMS:
        statuses = {}
        blacklist = {}
        try:
            statuses, n_files = load_statuses(reports_root / f"reports-{platform}")
            if not statuses:
                print(
                    f"WARNING: no report cases found under "
                    f"{reports_root / ('reports-' + platform)}",
                    file=sys.stderr,
                )
            if module is not None:
                blacklist = load_blacklist(module, config_dir, platform)
        except Exception as e:  # noqa: BLE001 - never raise
            print(f"WARNING: failed to load {platform} data: {e}", file=sys.stderr)
        loaded[platform] = {"statuses": statuses, "blacklist": blacklist}

    platforms_block = {}
    stale_by_origin = {p: [] for p in PLATFORMS}
    for platform in PLATFORMS:
        other = OTHER[platform]
        try:
            records, status_counts, retest = build_stale(
                platform,
                loaded[platform]["statuses"],
                loaded[platform]["blacklist"],
                loaded[other]["blacklist"],
                loaded[other]["statuses"],
            )
        except Exception as e:  # noqa: BLE001 - never raise
            print(f"WARNING: failed to build {platform} stale set: {e}", file=sys.stderr)
            records, status_counts, retest = [], {}, []
        platforms_block[platform] = {
            "blacklist_total": len(loaded[platform]["blacklist"]),
            "status_counts": status_counts,
            "stale_total": len(records),
            "cases": records,
        }
        stale_by_origin[platform] = retest

    # Shard JSONs are written to retest-cases-<target>/ and the retest block /
    # GITHUB_OUTPUT keys are keyed by the SAME <target> (the platform the cases
    # are re-run ON). The retest-<target> job, the retest-cases-<target>
    # artifact, the retest-<target> category and the rta<target> report prefix
    # share this key, so a platform's guard, shard count and artifact always
    # describe the same case set.
    retest_block = {}
    for origin in PLATFORMS:
        target = OTHER[origin]
        try:
            retest_block[target] = write_retest_shards(target, stale_by_origin[origin])
        except Exception as e:  # noqa: BLE001 - never raise
            print(f"WARNING: failed to write {target} retest shards: {e}", file=sys.stderr)
            retest_block[target] = {"num_shards": 1, "count": len(stale_by_origin[origin])}

    aggregate = {
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": RULE,
        "platforms": platforms_block,
        "retest": retest_block,
    }

    json_path = Path(f"{args.output_prefix}.json")
    json_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown = render_collect_markdown(aggregate)
    md_path = Path(f"{args.output_prefix}.md")
    md_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nwritten: {json_path}, {md_path}")

    outputs = {}
    for target in PLATFORMS:
        block = retest_block[target]
        num_shards = block["num_shards"] or 1
        outputs[f"{target}_retest_count"] = block["count"]
        outputs[f"{target}_num_shards"] = num_shards
        outputs[f"{target}_shards_json"] = json.dumps(list(range(1, num_shards + 1)))
    write_github_output(outputs)

    print("retest summary: " + json.dumps(outputs))


def parse_collect_args(argv=None):
    parser = argparse.ArgumentParser(description="Collect stale blacklist cases")
    parser.add_argument("--reports-root", default=".", help="Root containing reports-a3/ reports-a5/")
    parser.add_argument("--config-dir", default=".github/config", help="Blacklist config directory")
    parser.add_argument("--output-prefix", default="passed-skipped", help="Output prefix for .json/.md")
    return parser.parse_args(argv)


# ==============================================================================
# report
# ==============================================================================


def render_report_markdown(aggregate, retest_statuses):
    """Render the retest outcome Markdown report."""
    lines = ["## Blacklist stale cases — cross-platform retest", ""]
    lines.append(f"- generated_at: {aggregate.get('generated_at', '')}")
    lines.append(f"- rule: {aggregate.get('rule', RULE)}")
    lines.append("")

    for platform in PLATFORMS:
        target = OTHER[platform]
        block = aggregate.get("platforms", {}).get(platform, {}) or {}
        cases = block.get("cases", []) or []
        statuses = retest_statuses.get(target, {})

        rows = []
        counter = collections.Counter()
        for case in cases:
            if case.get("target_already_listed"):
                outcome = NOT_RETESTED
                detail = case.get("target_status_if_known")
            else:
                outcome = statuses.get(case["nodeid"], MISSING)
                detail = None
            counter[outcome] += 1
            rows.append((case, outcome, detail))

        lines.append(f"### {platform} → {target} (retest on {target})")
        lines.append("")
        counts = " | ".join(
            f"{s}: {counter.get(s, 0)}"
            for s in REPORT_STATUSES + (NOT_RETESTED, MISSING)
        )
        failing = counter.get("failed", 0) + counter.get("error", 0) + counter.get("timeout", 0)
        lines.append(f"- total stale: {len(cases)} | {counts}")
        lines.append(
            f"- candidates to add to the {target} blacklist "
            f"(failed/error/timeout on {target}): {failing}"
        )
        lines.append("")
        lines.append(
            "| # | nodeid | origin | origin status | retest platform | retest status | known target status |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for idx, (case, outcome, detail) in enumerate(rows, 1):
            lines.append(
                f"| {idx} | `{_md_cell(case['nodeid'])}` | {case.get('origin_platform')} | "
                f"{case.get('origin_status')} | {target} | {outcome} | {_md_cell(detail)} |"
            )
        lines.append("")
    return "\n".join(lines)


def run_report(args):
    aggregate = {}
    try:
        aggregate = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARNING: could not read aggregate {args.input}: {e}", file=sys.stderr)

    reports_root = Path(args.reports_root)
    retest_statuses = {}
    for target in PLATFORMS:
        try:
            statuses, n_files = load_statuses(reports_root / f"retest-reports-{target}")
            if not statuses:
                print(
                    f"WARNING: no retest reports under "
                    f"{reports_root / ('retest-reports-' + target)}",
                    file=sys.stderr,
                )
        except Exception as e:  # noqa: BLE001 - never raise
            print(f"WARNING: failed to load retest-reports-{target}: {e}", file=sys.stderr)
            statuses = {}
        retest_statuses[target] = statuses

    markdown = render_report_markdown(aggregate, retest_statuses)
    output_path = Path(args.output)
    try:
        output_path.write_text(markdown, encoding="utf-8")
    except OSError as e:
        print(f"WARNING: could not write {output_path}: {e}", file=sys.stderr)

    print(markdown)
    print(f"\nwritten: {output_path}")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        try:
            with open(step_summary, "a", encoding="utf-8") as handle:
                handle.write(markdown + "\n")
        except OSError:
            pass


def parse_report_args(argv=None):
    parser = argparse.ArgumentParser(description="Render retest outcome report")
    parser.add_argument("--input", default="passed-skipped.json", help="Aggregate JSON from collect")
    parser.add_argument("--reports-root", default=".", help="Root containing retest-reports-a3/ retest-reports-a5/")
    parser.add_argument("--output", default="passed-skipped-retest.md", help="Output Markdown path")
    return parser.parse_args(argv)


# ==============================================================================
# entry point
# ==============================================================================


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("collect", help="Collect passed/skipped blacklist cases")
    sub.add_parser("report", help="Render the retest outcome report")
    args, rest = parser.parse_known_args(argv)

    if args.command == "collect":
        run_collect(parse_collect_args(rest))
    elif args.command == "report":
        run_report(parse_report_args(rest))


if __name__ == "__main__":
    main()
