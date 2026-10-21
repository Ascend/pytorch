#!/usr/bin/env python3
"""
Diff two collection output directories (baseline stdout channel vs shadow
JSONL channel).

Compares, per category:
  - total case counts          (cases_collection_summary.json)
  - per-file case counts and nodeid lists (<cat>_cases_by_file.jsonl)

When a file's case count matches between the two channels, nodeids are
compared positionally (collection order is an invariant of both channels)
and differences are reported as "changed" — the expected signature of the
newline-truncation fix (truncated fake nodeid -> full nodeid at the same
position). When counts diverge, set differences (added / removed) are
reported instead.

Output: markdown to stdout, appended to $GITHUB_STEP_SUMMARY when set.
Exit code is always 0 — this is a shadow-phase reporting tool; the diff is
judged by humans (plan stage-2 gate: the diff converges to the known
signature — e.g. test_utils_config_module.py's newline names becoming
complete — for 2-3 consecutive nights).

Usage:
    python3 diff_collection.py --old collect-a3 --new collect-shadow
"""

import argparse
import json
import os
import sys
from pathlib import Path

MAX_DIFFS_PER_FILE = 20
MAX_FILES_IN_DETAIL = 20


def load_summary(dir_path: Path):
    p = dir_path / "cases_collection_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_by_file(dir_path: Path, category: str):
    """Parse <category>_cases_by_file.jsonl into {file_path: [nodeids]}.

    Line 1 is the summary record; lines 2+ carry the per-file case lists.
    Lines are split on "\\n" only (records are pure-ASCII JSON).
    """
    p = dir_path / f"{category}_cases_by_file.jsonl"
    if not p.exists():
        return None
    try:
        lines = [l for l in p.read_text(encoding="utf-8").split("\n") if l]
        files = {}
        for line in lines[1:]:
            obj = json.loads(line)
            if not isinstance(obj, dict) or "file_path" not in obj or "cases" not in obj:
                raise ValueError(f"malformed record: {line[:80]!r}")
            files[obj["file_path"]] = obj["cases"]
        return files
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as e:
        # Degrade to "side unreadable" (empty dict at the caller), never a
        # crash: this tool's contract is exit code 0 so the shadow-diff
        # step always renders a report.
        print(f"WARNING: unreadable {p.name}: {e}", file=sys.stderr)
        return None


def diff_file(old_cases, new_cases):
    """Describe the difference between two ordered nodeid lists (or None)."""
    if old_cases == new_cases:
        return None
    result = {"old_count": len(old_cases), "new_count": len(new_cases)}
    if len(old_cases) == len(new_cases):
        result["changed"] = [
            (i, o, n)
            for i, (o, n) in enumerate(zip(old_cases, new_cases))
            if o != n
        ]
    else:
        old_set, new_set = set(old_cases), set(new_cases)
        result["removed"] = sorted(old_set - new_set)
        result["added"] = sorted(new_set - old_set)
    return result


def fmt(nodeid):
    """One-line, control-character-safe rendering for markdown tables.

    Every non-printable character (NEL, U+2028, ...) is shown as a visible
    \\uXXXX escape so hazardous names cannot hide in the diff table.
    Backticks are escaped the same way — a literal backtick would
    terminate the markdown code span mid-row and mangle the table.
    """
    escaped = nodeid.replace("|", "\\|").replace("\n", "\\n")
    return "".join(
        c if (c.isprintable() and c != "`") else f"\\u{ord(c):04x}"
        for c in escaped
    )


def main():
    ap = argparse.ArgumentParser(description="Diff two collection outputs")
    ap.add_argument("--old", required=True,
                    help="baseline cases-shards dir (stdout channel)")
    ap.add_argument("--new", required=True,
                    help="shadow cases-shards dir (JSONL channel)")
    ap.add_argument("--label-old", default="stdout (baseline)")
    ap.add_argument("--label-new", default="JSONL (shadow)")
    args = ap.parse_args()

    old_dir, new_dir = Path(args.old), Path(args.new)
    out = []

    def emit(s=""):
        out.append(s)

    old_sum, new_sum = load_summary(old_dir), load_summary(new_dir)

    emit(f"# Collection channel diff: `{args.label_old}` vs `{args.label_new}`")
    emit()

    if old_sum is None or new_sum is None:
        missing = []
        if old_sum is None:
            missing.append(f"{args.old} (missing or unreadable summary)")
        if new_sum is None:
            missing.append(f"{args.new} (missing or unreadable summary)")
        emit(f"**Cannot diff** — missing inputs: {', '.join(missing)}. "
             "Check that both collect jobs succeeded and their artifacts "
             "were downloaded.")
        print("\n".join(out))
        _append_step_summary(out)
        return

    cats = sorted(
        set(old_sum.get("categories", {})) | set(new_sum.get("categories", {}))
    )

    # ---- per-category totals ----
    # A category missing on one side counts as 0 in the totals: skipping it
    # would let a vanished category read as "converged" (a 50-case loss
    # must surface as total -50, not as +0).
    emit("| category | old cases | new cases | Δ |")
    emit("|---|---:|---:|---:|")
    total_old = total_new = 0
    missing_cats = []
    for cat in cats:
        o = old_sum.get("categories", {}).get(cat, {}).get("total_cases")
        n = new_sum.get("categories", {}).get(cat, {}).get("total_cases")
        if o is None or n is None:
            missing_cats.append(cat)
        o_val = o if o is not None else 0
        n_val = n if n is not None else 0
        total_old += o_val
        total_new += n_val
        o_cell = str(o) if o is not None else "*(missing → 0)*"
        n_cell = str(n) if n is not None else "*(missing → 0)*"
        emit(f"| {cat} | {o_cell} | {n_cell} | {n_val - o_val:+d} |")
    emit(f"| **total** | **{total_old}** | **{total_new}** | "
         f"**{total_new - total_old:+d}** |")
    emit()

    # ---- per-file diffs ----
    files_with_diffs = []
    only_old_files, only_new_files = [], []

    for cat in cats:
        old_files = load_by_file(old_dir, cat) or {}
        new_files = load_by_file(new_dir, cat) or {}
        for fp in sorted(set(old_files) | set(new_files)):
            if fp not in new_files:
                only_old_files.append((cat, fp, len(old_files[fp])))
                continue
            if fp not in old_files:
                only_new_files.append((cat, fp, len(new_files[fp])))
                continue
            d = diff_file(old_files[fp], new_files[fp])
            if d:
                d["category"] = cat
                d["file"] = fp
                files_with_diffs.append(d)

    emit(f"Files with nodeid differences: **{len(files_with_diffs)}** "
         f"(positional `changed` entries are the expected signature of the "
         f"truncation fix; `added`/`removed` need investigation)")
    emit()

    for d in files_with_diffs[:MAX_FILES_IN_DETAIL]:
        emit(f"### {d['file']} ({d['category']}) — "
             f"{d['old_count']} → {d['new_count']} cases")
        if "changed" in d:
            emit(f"{len(d['changed'])} changed (same position, different string):")
            emit("| # | old | new |")
            emit("|---|---|---|")
            for i, o, n in d["changed"][:MAX_DIFFS_PER_FILE]:
                emit(f"| {i} | `{fmt(o)}` | `{fmt(n)}` |")
            if len(d["changed"]) > MAX_DIFFS_PER_FILE:
                emit(f"| ... | {len(d['changed']) - MAX_DIFFS_PER_FILE} more | |")
        else:
            for label, key in (("removed", "removed"), ("added", "added")):
                items = d.get(key, [])
                if items:
                    emit(f"{len(items)} {label}:")
                    for n in items[:MAX_DIFFS_PER_FILE]:
                        emit(f"- `{fmt(n)}`")
                    if len(items) > MAX_DIFFS_PER_FILE:
                        emit(f"- ... and {len(items) - MAX_DIFFS_PER_FILE} more")
        emit()
    if len(files_with_diffs) > MAX_FILES_IN_DETAIL:
        emit(f"... and {len(files_with_diffs) - MAX_FILES_IN_DETAIL} more files "
             f"with differences")
        emit()

    if only_old_files or only_new_files:
        emit("### Files present in only one channel")
        for cat, fp, cnt in only_old_files[:MAX_FILES_IN_DETAIL]:
            emit(f"- only in old ({cat}): {fp} ({cnt} cases)")
        for cat, fp, cnt in only_new_files[:MAX_FILES_IN_DETAIL]:
            emit(f"- only in new ({cat}): {fp} ({cnt} cases)")
        emit()

    # ---- shadow channel extras ----
    nr_path = new_dir / "nodeid_report.json"
    if nr_path.exists():
        try:
            nr = json.loads(nr_path.read_text(encoding="utf-8"))
            emit("### Shadow channel report (nodeid_report.json)")
            emit(f"- unselectable names (`::` in collector chain): "
                 f"**{nr.get('unselectable_count', '?')}**")
            emit(f"- hazardous names (linebreak / edge whitespace): "
                 f"**{nr.get('hazard_count', '?')}**")
            emit()
        except (json.JSONDecodeError, OSError):
            pass

    # ---- verdict line ----
    changed_total = sum(len(d.get("changed", [])) for d in files_with_diffs)
    added_total = sum(len(d.get("added", [])) for d in files_with_diffs)
    removed_total = sum(len(d.get("removed", [])) for d in files_with_diffs)
    if missing_cats:
        emit(f"**WARNING: {len(missing_cats)} category(ies) present on only one side: "
             f"{', '.join(missing_cats)} — counted as 0 in the totals above. "
             f"This is NOT convergence; check both collect jobs' configs and checkouts.**")
        emit()
    emit("---")
    emit(f"**Totals**: {changed_total} changed, {added_total} added, "
         f"{removed_total} removed; total cases {total_old} → {total_new}. "
         f"Stage-2 gate: diff must converge to the known signature "
         f"(changed-only, no added/removed, no missing categories) for 2-3 "
         f"consecutive nights.")

    print("\n".join(out))
    _append_step_summary(out)


def _append_step_summary(lines):
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
