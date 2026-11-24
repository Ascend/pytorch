#!/usr/bin/env python3
"""AI 失败用例分析的廉价门禁 + 失败数据预处理脚本。

在 nightly a3/a5 末尾的 ai-failure-analysis job 里最先执行（无模型调用）：
  1. 解析目标 run（run URL / job URL / 裸 run id）；
  2. 下载 npu-full-test-summary 制品，统计 failed/error/timeout 用例；
  3. 按基础设施错误签名过滤——job 级/基础设施失败一律跳过 AI 分析；
  4. 有非 infra 失败用例时，下载失败分片报告并解压用例日志，产出 failures.json。

用法:
  python3 fetch_test_failures.py --run-url <RUN_URL|RUN_ID> [--workdir failure-data]
      [--repo Ascend/pytorch] [--max-cases 30] [--max-msg-chars 2000]

产物（写入 --workdir）:
  gate.json       门禁结论（status/reason/counts/run 信息），workflow 读取它写 Summary
  failures.json   非 infra 失败用例详情（供 AI agent 起步）+ infra 用例清单
  summary/        npu-full-test-summary 制品
  reports-<sh>/   失败分片 test-reports 制品（含解压后的 cases_logs）

退出码:
  0 = 存在非 infra 失败用例，执行 AI 分析
  3 = 无 failed/error/timeout 用例（job 失败但用例全绿等），跳过
  4 = 无 npu-full-test-summary 制品（report/collect 等 job 级失败），跳过
  5 = 失败用例全部命中基础设施签名，跳过
  1 = 脚本自身错误（网络/解析异常等，让 job 显式失败）

依赖: gh CLI（GH_TOKEN 已认证）+ python3 标准库。
蓝本: ~/.claude/skills/pr-skip-list-workflow/scripts/fetch_failures.py。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tarfile

FAILED_STATES = ("failed", "error", "errors", "timeout")

# 基础设施错误签名：命中的用例不交给 AI 分析。与
# github-pipline-analy/scripts/block_cases.py 的 INFRA_PATTERNS 保持一致，
# 并补充 NPU 环境已知指纹（HCCL EI0006 建链失败、runner 失联措辞）。
INFRA_PATTERNS = re.compile(
    r"disk quota"
    r"|no space left"
    r"|runner (offline|lost)"
    r"|self-hosted runner"
    r"|connection (reset|refused|timed out)"
    r"|internal server error"
    r"|http 50[234]"
    r"|EI0006"
    r"|the runner has received a shutdown signal"
    r"|lost communication with the server"
    r"|cann't get valid.*device.*from.*cluster",
    re.IGNORECASE,
)

# 每个用例的 message 截断长度（failures.json 内），控制 agent 输入体积
DEFAULT_MAX_MSG_CHARS = 2000
# infra 清单在 failures.json 里的截断条数（nodeid 全量计数始终保留）
MAX_INFRA_LISTED = 200


def sh(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r


def gh_api(path, repo):
    out = sh(["gh", "api", f"repos/{repo}/{path}"])
    return json.loads(out.stdout)


def parse_run_id(target, repo):
    m = re.search(r"/runs/(\d+)", target)
    if m:
        return m.group(1), None
    m = re.search(r"/job/(\d+)", target)
    if m:
        job = gh_api(f"actions/jobs/{m.group(1)}", repo)
        return str(job["run_id"]), job
    if target.isdigit():
        return target, None
    raise SystemExit(f"cannot parse run/job id from: {target}")


def download_artifact(run_id, repo, name, dest):
    os.makedirs(dest, exist_ok=True)
    sh(["gh", "run", "download", run_id, "--repo", repo, "--name", name, "--dir", dest])


def write_gate(workdir, payload):
    payload.setdefault("status", "skipped")
    with open(os.path.join(workdir, "gate.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-url", required=True,
                    help="流水线 run URL / job URL / 裸 run id")
    ap.add_argument("--workdir", default="failure-data")
    ap.add_argument("--repo", default="Ascend/pytorch")
    ap.add_argument("--max-cases", type=int, default=30,
                    help="failures.json 中保留完整详情的非 infra 用例上限")
    ap.add_argument("--max-msg-chars", type=int, default=DEFAULT_MAX_MSG_CHARS)
    args = ap.parse_args()

    run_id, job = parse_run_id(args.run_url, args.repo)
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)

    try:
        run = gh_api(f"actions/runs/{run_id}", args.repo)
        arts = gh_api(f"actions/runs/{run_id}/artifacts?per_page=100", args.repo)
    except SystemExit:
        raise
    art_names = {a["name"] for a in arts.get("artifacts", [])}

    info = {
        "run_id": int(run_id),
        "name": run.get("name"),
        "event": run.get("event"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "created_at": run.get("created_at"),
        "head_sha": run.get("head_sha"),
        "html_url": run.get("html_url"),
    }
    print(f"Run:      {info['name']} [{run_id}]")
    print(f"Event:    {info['event']}  Conclusion: {info['conclusion']}")
    print(f"Created:  {info['created_at']}  torch_npu@{(info['head_sha'] or '')[:12]}")
    if job:
        print(f"Input job: {job['name']} -> {job['conclusion']}")

    # ---- 门禁第一跳：报告制品必须存在（report job 成功的标志） ----
    if "npu-full-test-summary" not in art_names:
        reason = ("npu-full-test-summary artifact missing: report job did not "
                  "succeed (job-level / infrastructure failure), AI analysis skipped")
        print("GATE: " + reason)
        write_gate(workdir, {"skip_code": 4, "reason": reason, "run": info})
        sys.exit(4)

    dest = os.path.join(workdir, "summary")
    download_artifact(run_id, args.repo, "npu-full-test-summary", dest)
    summary_json = os.path.join(dest, "npu-full-test-summary.json")
    if not os.path.exists(summary_json):
        reason = "summary artifact downloaded but npu-full-test-summary.json missing"
        print("GATE: " + reason)
        write_gate(workdir, {"skip_code": 4, "reason": reason, "run": info})
        sys.exit(4)

    summary = json.load(open(summary_json, encoding="utf-8"))
    totals = summary.get("totals", {}) or {}
    info.update({
        "pytorch_version": summary.get("pytorch_version"),
        "torch_npu_short": summary.get("torch_npu_short"),
        "docker_image": summary.get("docker_image"),
        "runner": summary.get("runner"),
    })
    counts = {
        "failed": int(totals.get("failed", 0) or 0),
        "errors": int(totals.get("errors", 0) or 0),
        "timeout": int(totals.get("timeout", 0) or 0),
    }
    print(f"Versions: pytorch@{info.get('pytorch_version')} torch_npu@{info.get('torch_npu_short')}")
    print(f"Totals:   {json.dumps(totals)}")

    # ---- 门禁第二跳：必须有 failed/error/timeout 用例 ----
    failing_shards = []
    for s in summary.get("shards", []):
        if s.get("failed", 0) or s.get("errors", 0) or s.get("timeout", 0):
            failing_shards.append(s["shard"])
    if counts["failed"] + counts["errors"] + counts["timeout"] == 0 and not failing_shards:
        reason = ("no failed/error/timeout cases (either all green, or job-level "
                  "failure with cases green e.g. SIGABRT shutdown / runner loss), "
                  "AI analysis skipped")
        print("GATE: " + reason)
        write_gate(workdir, {"skip_code": 3, "reason": reason, "run": info,
                             "totals": totals, "counts": counts})
        sys.exit(3)

    # ---- 下载失败分片报告，提取用例并做 infra 分类 ----
    cases_all = []  # (shard, case dict, log_file)
    missing_shards = []
    for shard in failing_shards:
        art = f"test-reports-{shard}"
        if art not in art_names:
            print(f"WARNING: artifact {art} not found, skip shard {shard}")
            missing_shards.append(shard)
            continue
        sdest = os.path.join(workdir, f"reports-{shard}")
        download_artifact(run_id, args.repo, art, sdest)
        cases_file = os.path.join(sdest, f"shard_{shard}_cases.json")
        if not os.path.exists(cases_file):
            print(f"WARNING: {cases_file} not found, skip shard {shard}")
            continue
        data = json.load(open(cases_file, encoding="utf-8"))
        for c in data.get("cases", []):
            if c.get("status") not in FAILED_STATES:
                continue
            idx = c.get("case_idx")
            log_file = None
            for root, _, files in os.walk(sdest):
                for f in files:
                    if f.startswith(f"{shard}_{idx}_") and f.endswith(".log"):
                        log_file = os.path.join(root, f)
                        break
                if log_file:
                    break
            cases_all.append((shard, c, log_file))

    non_infra = [(sh, c, lg) for (sh, c, lg) in cases_all
                 if not INFRA_PATTERNS.search(c.get("message", "") or "")]
    infra = [(sh, c, lg) for (sh, c, lg) in cases_all
             if INFRA_PATTERNS.search(c.get("message", "") or "")]

    def base_rec(sh, c, lg):
        return {
            "nodeid": c["nodeid"],
            "status": c["status"],
            "duration": c.get("duration"),
            "returncode": c.get("returncode"),
            "shard": sh,
            "case_idx": c.get("case_idx"),
            "log_file": lg,
        }

    # ---- 门禁第三跳：失败用例不能全部是 infra 签名 ----
    if not non_infra:
        reason = (f"all {len(cases_all)} failed/error/timeout cases match "
                  "infrastructure signatures, AI analysis skipped")
        print("GATE: " + reason)
        write_gate(workdir, {"skip_code": 5, "reason": reason, "run": info,
                             "totals": totals, "counts": counts,
                             "cases_total": len(cases_all), "cases_infra": len(infra)})
        sys.exit(5)

    # ---- 为包含非 infra 用例的分片解压 cases_logs（infra-only 分片不解压） ----
    non_infra_shards = {sh for (sh, _, _) in non_infra}
    for shard in sorted(non_infra_shards):
        sdest = os.path.join(workdir, f"reports-{shard}")
        logs_tar = os.path.join(sdest, "cases_logs.tar.gz")
        logs_dir = os.path.join(sdest, "cases_logs")
        if os.path.exists(logs_tar) and not os.path.exists(logs_dir):
            try:
                with tarfile.open(logs_tar) as tf:
                    tf.extractall(sdest)  # tar 内可能嵌套 cases_logs/ 目录
            except Exception as e:
                print(f"WARNING: extract cases_logs for shard {shard} failed: {e}")

    # 重新解析日志路径（解压之后）
    for i, (sh, c, lg) in enumerate(non_infra):
        if lg:
            continue
        idx = c.get("case_idx")
        sdest = os.path.join(workdir, f"reports-{sh}")
        for root, _, files in os.walk(sdest):
            hit = next((f for f in files
                        if f.startswith(f"{sh}_{idx}_") and f.endswith(".log")), None)
            if hit:
                non_infra[i] = (sh, c, os.path.join(root, hit))
                break

    # ---- 生成 failures.json（详情截断，nodeid 全量保留） ----
    def detail_rec(sh, c, lg):
        rec = base_rec(sh, c, lg)
        msg = (c.get("message", "") or "").strip()
        if len(msg) > args.max_msg_chars:
            msg = msg[: args.max_msg_chars] + "..."
        rec["message"] = msg
        rec["file"] = c.get("file")
        return rec

    capped = non_infra[: args.max_cases]
    cases_out = [detail_rec(sh, c, lg) for (sh, c, lg) in capped]
    if len(non_infra) > args.max_cases:
        print(f"NOTE: {len(non_infra)} non-infra cases, detail capped to "
              f"{args.max_cases} (full nodeid list kept in all_non_infra_nodeids)")

    infra_out = [base_rec(sh, c, lg) for (sh, c, lg) in infra[:MAX_INFRA_LISTED]]

    failures = {
        "run": info,
        "totals": totals,
        "counts": counts,
        "non_infra_count": len(non_infra),
        "infra_count": len(infra),
        "analyzed_detail_count": len(cases_out),
        "all_non_infra_nodeids": [c["nodeid"] for (_, c, _) in non_infra],
        "infra_cases": infra_out,
        "missing_shard_artifacts": missing_shards,
        "cases": cases_out,
    }
    with open(os.path.join(workdir, "failures.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=1)

    print(f"\n===== failed/error/timeout cases: {len(cases_all)} "
          f"(non-infra: {len(non_infra)}, infra: {len(infra)}) =====")
    for r in cases_out:
        print(f"\n[{r['shard']} #{r['case_idx']}] {r['status']} "
              f"({r['duration']:.1f}s) {r['nodeid']}")
        if r["message"]:
            print("  msg: " + r["message"][:400].replace("\n", "\n       "))
        if r["log_file"]:
            print(f"  log: {r['log_file']}")

    reason = (f"{len(non_infra)} non-infrastructure failed/error/timeout cases "
              f"(infra-classified: {len(infra)})")
    write_gate(workdir, {"status": "analyze", "skip_code": None, "reason": reason,
                         "run": info, "totals": totals, "counts": counts,
                         "cases_total": len(cases_all),
                         "cases_non_infra": len(non_infra),
                         "cases_infra": len(infra)})
    print(f"\nGATE: analyze ({reason}); data ready: "
          f"{os.path.join(workdir, 'failures.json')}")
    sys.exit(0)


if __name__ == "__main__":
    main()
