#!/usr/bin/env python3
"""
Send a Feishu group-bot notification when the consolidated NPU test report
contains failed, errored, or timed-out cases.

Reads npu-full-test-summary.json (and optionally skipped_cases.json) from the
report job workspace, builds a plain-text summary, and posts it to the
FEISHU_WEBHOOK_URL webhook. Exits 0 silently when there is nothing to report
or when the webhook is not configured, so notification problems never fail
the report job.
"""

import argparse
import json
import os
import sys
import urllib.request

WORKFLOW_FILE_MAP = {
    "Nightly Build and Test": "nightly.yml",
    "Nightly CPU Full Test": "nightly-cpu.yml",
    "PyTorch CI Trigger PR": "pytorch_ci_trigger_pr.yml",
    "PyTorch CI Trigger Push": "pytorch_ci_trigger_push.yml",
}

CATEGORY_DISPLAY = {
    "distributed": "dist",
    "core": "core",
    "tensor": "tensor",
    "graph": "graph",
    "others": "others",
    "regular": "reg",
    "custom": "custom",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send Feishu failure notification for the NPU CI test report"
    )
    parser.add_argument("--report-json", required=True, help="Path to npu-full-test-summary.json")
    parser.add_argument(
        "--skipped-cases", default="skipped_cases.json", help="Path to skipped_cases.json"
    )
    parser.add_argument("--webhook", help="Feishu webhook URL (defaults to $FEISHU_WEBHOOK_URL)")
    return parser.parse_args()


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        print(f"[feishu] WARN: cannot load {path}: {e}")
        return None


def _as_int(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def workflow_display():
    name = os.environ.get("GITHUB_WORKFLOW", "unknown")
    workflow_file = WORKFLOW_FILE_MAP.get(name)
    if workflow_file:
        return f"{name} ({workflow_file})"
    return name


def build_distribution(report):
    per_category = {}
    for shard in report.get("shards", []):
        shard_type = str(shard.get("shard_type", ""))
        entry = per_category.setdefault(
            shard_type, {"failed": 0, "errors": 0, "timeout": 0}
        )
        for key in ("failed", "errors", "timeout"):
            entry[key] += _as_int(shard.get(key))

    parts = []
    for shard_type, counts in per_category.items():
        sub = []
        if counts["failed"]:
            sub.append(f"失败{counts['failed']}")
        if counts["errors"]:
            sub.append(f"错误{counts['errors']}")
        if counts["timeout"]:
            sub.append(f"超时{counts['timeout']}")
        if sub:
            display = CATEGORY_DISPLAY.get(shard_type, shard_type)
            parts.append(f"{display}({'/'.join(sub)})")
    return " | ".join(parts) if parts else "-"


def build_message(report, skipped_total):
    totals = report.get("totals", {})
    total = _as_int(totals.get("total"))
    passed = _as_int(totals.get("passed"))
    failed = _as_int(totals.get("failed"))
    errors = _as_int(totals.get("errors"))
    timeout = _as_int(totals.get("timeout"))
    skipped = _as_int(totals.get("skipped"))

    pytorch_short = str(report.get("pytorch_version", "unknown"))
    torch_npu_short = str(report.get("torch_npu_short", "unknown"))

    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if repo and run_id:
        run_url = f"{server}/{repo}/actions/runs/{run_id}"
    else:
        run_url = "unavailable"

    lines = [
        "【NPU CI 测试失败告警】",
        f"触发Workflow: {workflow_display()}",
        f"Run名称: {os.environ.get('GITHUB_RUN_NAME', 'unknown')}",
        f"版本: pytorch {pytorch_short} / torch_npu {torch_npu_short}",
        f"总用例: {total} | 通过: {passed} | 失败: {failed} | "
        f"Error: {errors} | 超时: {timeout} | 跳过: {skipped}",
        f"黑名单命中过滤: {skipped_total}",
        f"失败分布: {build_distribution(report)}",
        f"详情: {run_url}",
    ]
    return "\n".join(lines)


def send(webhook, text):
    payload = json.dumps({"msg_type": "text", "content": {"text": text}}).encode("utf-8")
    request = urllib.request.Request(
        webhook,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = json.loads(response.read().decode("utf-8"))
            code = body.get("code", body.get("StatusCode", 0))
            if code != 0:
                print(f"[feishu] WARN: webhook rejected message: {body}")
            else:
                status = body.get("msg", body.get("StatusMessage", "ok"))
                print(f"[feishu] notification sent ({status})")
    except Exception as e:
        print(f"[feishu] WARN: failed to send notification: {e}")


def main():
    args = parse_args()

    report = load_json(args.report_json)
    if report is None:
        print("[feishu] no report data, skipping notification")
        return 0

    totals = report.get("totals", {})
    failed = _as_int(totals.get("failed"))
    errors = _as_int(totals.get("errors"))
    timeout = _as_int(totals.get("timeout"))
    if failed + errors + timeout == 0:
        print("[feishu] no failed/error/timeout cases, skipping notification")
        return 0

    skipped_data = load_json(args.skipped_cases) if args.skipped_cases else None
    skipped_total = _as_int((skipped_data or {}).get("total_skipped"))

    webhook = args.webhook or os.environ.get("FEISHU_WEBHOOK_URL", "")
    if not webhook:
        print("[feishu] WARN: FEISHU_WEBHOOK_URL not configured, skipping notification")
        return 0

    message = build_message(report, skipped_total)
    print("[feishu] message preview:")
    print(message)
    send(webhook, message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
