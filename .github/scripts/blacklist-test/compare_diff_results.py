#!/usr/bin/env python3
"""Compare cross-platform diff results and propose blacklist sync candidates.

For every nodeid that exists in only one platform's blacklist:

    * read the declared entry (category/reason/issue + which file/group holds it)
    * read the result of the same nodeid on BOTH platforms (the diff shards run
      each set on a3 and a5, so both sides are available in this run)
    * a sync candidate is produced only when
        - the owning platform still fails (otherwise the entry is stale), and
        - the other platform also fails, and
        - its failure log matches the declared reason:
            * ``acl not supported: <aclnnXxx>``  -> log must contain
              ``call <aclnnXxx> failed`` (logger form used by op-plugin);
            * any other reason -> the reason text must appear in the log
              (case-insensitive, small normalisation applied).

Nothing is written back to the repository: the script only emits
``blacklist-sync-candidates.json`` plus a Markdown report.
"""

import argparse
import collections
import datetime
import json
import re
from pathlib import Path

BLACKLIST_FILES = [
    ("disabled", "disabled_testcases_{platform}_all.json", None),
    ("aclnn", "unsupported_aclnn_testcases.json", ("common", "{platform}")),
    ("dtype", "unsupport_dtype_testcases.json", ("common", "{platform}")),
]

FAILED_STATUSES = ("failed", "error")
ACL_PREFIX = "acl not supported:"
DT_SUFFIX = re.compile(r"\s*\(DT_[A-Za-z0-9_]+\)\s*$")
RUNTIME_PREFIX = re.compile(r"^(runtimeerror|assertionerror|exception):\s*", re.I)


def load_entries(config_dir: Path, platform: str):
    """nodeid -> {file, group, category, reason, issue}; first match wins."""
    entries = {}
    for label, tpl, groups in BLACKLIST_FILES:
        path = config_dir / tpl.format(platform=platform)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if groups is None:
            buckets = [(None, data)]
        else:
            buckets = [
                (g.format(platform=platform), data.get(g.format(platform=platform), {}))
                for g in groups
                if isinstance(data.get(g.format(platform=platform)), dict)
            ]
        for group, bucket in buckets:
            for nodeid, meta in bucket.items():
                if not isinstance(meta, dict) or nodeid in entries:
                    continue
                entries[nodeid] = {
                    "file": tpl.format(platform=platform),
                    "group": group,
                    "category": str(meta.get("category") or ""),
                    "reason": str(meta.get("reason") or ""),
                    "issue": str(meta.get("issue") or ""),
                }
    return entries


def load_results(reports_dir: Path):
    """nodeid -> {status, returncode, message}; last shard wins."""
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
                    "returncode": case.get("returncode"),
                    "message": case.get("message") or "",
                }
    return results


def _snippet(text: str, needle: str, width: int = 60) -> str:
    idx = text.lower().find(needle.lower())
    if idx < 0:
        return text.replace("\n", " ")[:160]
    start = max(0, idx - width)
    end = min(len(text), idx + len(needle) + width * 2)
    return re.sub(r"\s+", " ", text[start:end]).strip()[:240]


def match_reason(reason: str, message: str):
    """Return (mode, evidence) when the log supports ``reason``, else ("", "")."""
    reason = reason.strip()
    if not reason or not message:
        return "", ""
    low = message.lower()

    if reason.lower().startswith(ACL_PREFIX):
        op = reason[len(ACL_PREFIX):].strip()
        op = DT_SUFFIX.sub("", op).strip()
        if op.lower().startswith("aclnn"):
            needle = f"call {op.lower()} failed"
            if needle in low:
                return "call_op_failed", _snippet(message, needle)
            return "", ""
        # Non-aclnn names (log1p / eye / Tensor.log_normal_ ...) cannot produce
        # the `call <op> failed` form.  Reported separately, never a candidate.
        if op.lower() in low:
            return "weak_op_name", _snippet(message, op)
        return "", ""

    if reason.lower() in low:
        return "reason_text", _snippet(message, reason)

    normalised = RUNTIME_PREFIX.sub("", reason.lower()).rstrip(".")
    if normalised and normalised in low:
        return "reason_text_normalised", _snippet(message, normalised)
    return "", ""


def suggest_target(entry, other_platform: str):
    """Where the candidate would go on the other platform's blacklist."""
    if entry["file"].startswith("disabled_testcases"):
        return {"file": f"disabled_testcases_{other_platform}_all.json", "group": None}
    if entry["file"].startswith("unsupported_aclnn"):
        return {"file": "unsupported_aclnn_testcases.json",
                "group": entry["group"] if entry["group"] == "common" else other_platform}
    return {"file": "unsupport_dtype_testcases.json",
            "group": entry["group"] if entry["group"] == "common" else other_platform}


def compare_set(set_name, owner_platform, other_platform, own_entries, other_nodeids,
                results, top_mismatch=20):
    """Compare one diff set; returns (counters, candidates, mismatches)."""
    counters = collections.Counter()
    candidates = []
    mismatches = []

    targets = sorted(n for n in own_entries if n not in other_nodeids)
    counters["total"] = len(targets)

    for nodeid in targets:
        entry = own_entries[nodeid]
        own = results[owner_platform].get(nodeid)
        other = results[other_platform].get(nodeid)

        if own is None:
            counters["owner_no_result"] += 1
        elif own["status"] not in FAILED_STATUSES:
            counters["owner_not_failed"] += 1
        else:
            counters["owner_failed"] += 1

        if other is None:
            counters["other_no_result"] += 1
            mismatches.append({"nodeid": nodeid, "verdict": "other_no_result",
                               "reason": entry["reason"]})
            continue
        if other["status"] == "passed":
            counters["other_passed"] += 1
            mismatches.append({"nodeid": nodeid, "verdict": "other_passed",
                               "reason": entry["reason"]})
            continue
        if other["status"] not in FAILED_STATUSES:
            counters["other_skipped_or_timeout"] += 1
            mismatches.append({"nodeid": nodeid, "verdict": f"other_{other['status']}",
                               "reason": entry["reason"]})
            continue

        counters["other_failed"] += 1
        mode, evidence = match_reason(entry["reason"], other["message"])
        if mode == "weak_op_name":
            counters["reason_unverifiable"] += 1
            mismatches.append({"nodeid": nodeid, "verdict": "reason_unverifiable",
                               "reason": entry["reason"], "match_mode": mode,
                               "evidence": evidence})
            continue
        if not mode:
            counters["reason_mismatch"] += 1
            mismatches.append({"nodeid": nodeid, "verdict": "reason_mismatch",
                               "reason": entry["reason"],
                               "evidence": (other["message"] or "").replace("\n", " ")[:200]})
            continue

        counters["candidate"] += 1
        candidates.append({
            "nodeid": nodeid,
            "set": set_name,
            "owner_platform": owner_platform,
            "target_platform": other_platform,
            "category": entry["category"],
            "reason": entry["reason"],
            "issue": entry["issue"],
            "source": {"file": entry["file"], "group": entry["group"]},
            "suggested_target": suggest_target(entry, other_platform),
            "match_mode": mode,
            "evidence": evidence,
            "owner_status": own["status"] if own else None,
            "target_status": other["status"],
        })

    return counters, candidates, mismatches


def render_markdown(sets_report, candidates, top):
    lines = ["## 跨平台黑名单同步候选", ""]
    lines.append(
        "> 判定规则：差异用例在双方都跑；本侧仍失败、对侧也失败，且对侧日志能支撑声明的 `reason`，"
        "才列为候选。`acl not supported: <aclnnXxx>` 要求日志出现 `call <aclnnXxx> failed`。"
        "本报告只产出候选，不修改仓库。"
    )
    lines.append("")

    lines.append("### 总览")
    lines.append("")
    lines.append(
        "| set | 差异数 | 本侧失败 | 对侧失败 | **候选** | 对侧通过 | 对侧跳过/超时 | reason 不符 | reason 不可校验 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for set_name, c in sets_report.items():
        lines.append(
            f"| {set_name} | {c['total']} | {c['owner_failed']} | {c['other_failed']} | "
            f"**{c['candidate']}** | {c['other_passed']} | {c['other_skipped_or_timeout']} | "
            f"{c['reason_mismatch']} | {c['reason_unverifiable']} |"
        )
    lines.append("")

    grouped = collections.Counter(
        (cand["reason"], cand["category"], cand["suggested_target"]["file"],
         cand["suggested_target"]["group"] or "-")
        for cand in candidates
    )
    lines.append(f"### 候选汇总（Top {top}）")
    lines.append("")
    lines.append("| reason | category | 条数 | 建议写入 |")
    lines.append("|---|---|---|---|")
    for (reason, category, target_file, group), count in grouped.most_common(top):
        lines.append(f"| {reason.replace('|', chr(92) + '|')} | {category} | {count} | {target_file} :: {group} |")
    if len(grouped) > top:
        rest = sum(count for _, count in grouped.most_common()[top:])
        lines.append(f"| 其余 {len(grouped) - top} 种（合计） | - | {rest} | - |")
    lines.append("")

    lines.append(f"### 候选明细（前 {top} 条）")
    lines.append("")
    lines.append("| nodeid | reason | 匹配方式 | 证据 |")
    lines.append("|---|---|---|---|")
    for cand in candidates[:top]:
        evidence = (cand["evidence"] or "").replace("|", chr(92) + "|")[:140]
        lines.append(
            f"| {cand['nodeid']} | {cand['reason'].replace('|', chr(92) + '|')} | "
            f"{cand['match_mode']} | {evidence} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", default="diff-reports",
                        help="Directory containing <set>-<platform>/ result dirs")
    parser.add_argument("--config-dir", default=".github/config")
    parser.add_argument("--output-prefix", default="blacklist-sync-candidates")
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    reports_root = Path(args.reports_root)
    config_dir = Path(args.config_dir)

    entries = {p: load_entries(config_dir, p) for p in ("a3", "a5")}
    results = {
        (s, p): load_results(reports_root / f"{s}-{p}")
        for s in ("a3only", "a5only")
        for p in ("a3", "a5")
    }

    plan = [
        ("a3only", "a3", "a5"),
        ("a5only", "a5", "a3"),
    ]

    summary = {}
    candidates = []
    mismatches = []
    for set_name, owner, other in plan:
        counters, cand, mism = compare_set(
            set_name, owner, other,
            entries[owner], set(entries[other]),
            {p: results[(set_name, p)] for p in ("a3", "a5")},
        )
        summary[set_name] = counters
        candidates.extend(cand)
        mismatches.extend(mism)

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "rule": "owner failed + other failed + other log supports the declared reason",
        "sets": {k: dict(v) for k, v in summary.items()},
        "totals": {
            "candidates": len(candidates),
            "mismatches": len(mismatches),
        },
        "candidates": candidates,
        "mismatches": mismatches,
    }

    json_path = Path(f"{args.output_prefix}.json")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown = render_markdown(summary, candidates, args.top)
    md_path = Path(f"{args.output_prefix}.md")
    md_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nwritten: {json_path}, {md_path}")

    step_summary = __import__("os").environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")


if __name__ == "__main__":
    main()
