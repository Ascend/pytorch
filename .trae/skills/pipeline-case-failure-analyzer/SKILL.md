---
name: pipeline-case-failure-analyzer
description: "分析 Ascend/pytorch GitHub Actions 流水线中测试用例失败的根因。输入一个失败的流水线 run URL，定位哪些用例失败、为什么失败、是哪个变更引入的回归。覆盖 CI 夜间自动化模式（读取预处理好的 failures.json，按预算约束产出报告文件）与人工交互模式（从零采集）。触发词：分析流水线失败、用例失败根因、pipeline regression、CI 用例回归、定位引入问题的 PR。"
---

# Pipeline Case Failure Analyzer

分析 Ascend/pytorch CI 流水线中测试用例失败的根因，定位引入回归的 PR 或代码变更。

## 适用场景

- 用户提供一个失败的流水线 run URL（`https://github.com/Ascend/pytorch/actions/runs/XXXXX`）
- 用户需要定位失败用例的根因
- 用户需要对比两个流水线找出回归
- 用户需要确认引入问题的具体 PR 或代码变更

## 运行模式

两种模式，先判断当前处于哪种：

1. **夜间自动化模式（CI job 中）**：流水线预处理脚本
   （`.github/scripts/fetch_test_failures.py`）已生成 `failure-data/failures.json`
   （含 nodeid/status/duration/message/log_file/infra 分类）并解压好用例日志。
   **必须从"阶段 0"起步**，遵守"夜间自动化模式约束"节的预算。
2. **人工交互模式（无预处理数据）**：从"阶段 1"开始自行采集。

## 夜间自动化模式约束（CI job 中必读）

- **只分析非基础设施失败用例**：`failures.json` 中 `cases` 数组即分析对象
  （infra 类已剔除到 `infra_cases`，只在报告"基础设施类"节列一行清单，不分析）。
- **GitHub API 一律用 `gh api`**（`GH_TOKEN` 已认证，5000 req/h）。
  **禁止裸 `curl https://api.github.com/...`**——CI runner 的共享 IP 会撞 60 req/h
  匿名限流，全部请求都会失败。
- **预算约束**（由 prompt 注入 `MAX_CASES` / `TIME_BUDGET`）：
  - 失败用例先按错误签名（异常类型+首行消息）聚类；深度分析 top `MAX_CASES` 条，
    其余在报告"失败概览"节按簇给一行归类；
  - CANN 日志制品（`ascend-cann-logs-*`，单个 20~115MB）**默认跳过**，仅当某用例
    根因必须算子级证据且所在分片制品 < 200MB 时才下载，报告中注明是否下载；
  - 历史对比限定最近 24h 内至多 2 条同 workflow run；
  - 不做无界 `git log`/全量 diff 浏览，diff 范围限定在失败用例直接相关的文件。
- **产出契约**：完整报告写入 prompt 指定的 `REPORT_PATH` 文件（结构见"产出格式"），
  最终回答只输出 ≤10 行结论摘要。报告是给人看的夜间报告：结论先行、每条结论
  带置信度（高/中/低）与依据。

## 分析工作流

按阶段依次执行。夜间自动化模式从阶段 0 起步（阶段 1/2 的采集已由预处理完成）。

### 阶段 0：读取预处理数据（夜间自动化模式）

```bash
# 门禁结论与计数
python3 -c "
import json
g = json.load(open('failure-data/gate.json'))
print(g['status'], g['reason'])
print('counts:', g.get('counts'))
"
# 失败用例详情（分析对象）
python3 -c "
import json
d = json.load(open('failure-data/failures.json'))
print('non-infra:', d['non_infra_count'], 'infra:', d['infra_count'])
for c in d['cases']:
    print(c['shard'], c['status'], c['nodeid'])
"
# 查看某条用例的完整 pytest 日志（含 STDOUT/STDERR/重试历史）
cat failure-data/reports-<sh>/cases_logs/<sh>_<idx>_<name>.log
```

### 阶段 1：流水线基本信息采集

```bash
gh api repos/Ascend/pytorch/actions/runs/<RUN_ID> --jq '
  {commit: .head_sha, event: .event, status: .status,
   conclusion: .conclusion, created: .created_at, name: .name}'
gh api "repos/Ascend/pytorch/actions/runs/<RUN_ID>/jobs?per_page=100" --jq '
  .jobs[] | "\(.name): \(.conclusion)"'
gh api repos/Ascend/pytorch/commits/<COMMIT_SHA> --jq '
  .commit.message, (.files[:20][] | "  \(.filename) (+\(.additions) -\(.deletions))")'
```

记录关键字段：`head_sha`（commit）、`event`、`name`（workflow 名称）、`created_at`。

### 阶段 2：下载测试报告并定位失败用例

人工模式下（或需要预处理之外的制品时）：

```bash
# 全局摘要
gh run download <RUN_ID> --repo Ascend/pytorch --name npu-full-test-summary --dir summary
python3 -c "
import json
data = json.load(open('summary/npu-full-test-summary.json'))
print('Totals:', json.dumps(data['totals']))
for s in data.get('shards', []):
    if s.get('failed', 0) or s.get('errors', 0) or s.get('timeout', 0):
        print(f\"Shard {s['shard']}: failed={s['failed']}, errors={s['errors']}\")"

# 失败分片报告（含 shard_<sh>_cases.json 与 cases_logs.tar.gz）
gh run download <RUN_ID> --repo Ascend/pytorch --name test-reports-<SHARD> --dir rep-<SHARD>
mkdir -p rep-<SHARD>/cases_logs && tar xzf rep-<SHARD>/cases_logs.tar.gz -C rep-<SHARD>
```

注意：分片 job 本身失败但其 `test-reports-<shard>` 制品缺失时（如 shutdown 竞态/
runner 失联），说明是 job 级失败而非用例失败，summary 里不会计入——不要臆测其用例状态。

### 阶段 3：获取对应 commit 的测试源码

**关键：必须基于失败流水线的 commit 获取测试代码，不能使用本地代码或 main 分支代码。**

```bash
gh api "repos/Ascend/pytorch/contents/test/<PATH>?ref=<COMMIT_SHA>" --jq '.content' \
  | base64 -d > test_file.py
# 或（大文件 / raw）：
curl -s "https://raw.githubusercontent.com/Ascend/pytorch/<COMMIT_SHA>/test/<PATH>.py"
```

阅读重点：失败的 assert 与期望值 vs 实际值；输入构造（空输入、零维、非连续等
边界）；reference 实现；dtype/device/shape 参数化。

### 阶段 4：查找历史流水线并对比

向前查找 24 小时内的同 workflow 流水线，确认失败用例之前是否通过。

```bash
gh api "repos/Ascend/pytorch/actions/workflows/<WORKFLOW_ID>/runs?per_page=10" \
  --jq '.workflow_runs[] | "\(.id) | \(.head_sha[:12]) | \(.created_at) | \(.conclusion)"'
# 下载候选 run 的分片报告后搜索目标 nodeid
gh run download <OLD_RUN_ID> --repo Ascend/pytorch --name test-reports-<N> --dir old_reg<N>
python3 -c "
import json, glob
targets = ['<失败用例关键词>']
for f in sorted(glob.glob('old_reg*/shard_*_cases.json')):
    for c in json.load(open(f)).get('cases', []):
        for t in targets:
            if t in c.get('nodeid', ''):
                print(f'{f}: {c[\"nodeid\"]} -> {c[\"status\"]}')"
```

三种对比结果：
1. **之前通过，现在失败** → 回归，进入阶段 5
2. **之前不存在** → 可能是新参数化用例，查上游 pytorch 是否新增
3. **之前也失败** → 存量问题，查白名单/历史 issue

### 阶段 5：分析两个 commit 之间的代码变更

```bash
# diff 概览
gh api repos/Ascend/pytorch/compare/<OLD_SHA>...<NEW_SHA> --jq '
  {total_commits, status, files: (.files | length)}'
# 子模块指针比对（op-plugin/CANN 变更常是根因）
gh api "repos/Ascend/pytorch/contents/third_party/op-plugin?ref=<SHA>" --jq '.sha'
# 本地 git（runner 已 checkout 本仓库时）
git fetch origin <SHA> && git diff <OLD>..<NEW> -w --stat -- torch_npu/csrc/aten/
```

系统性排查顺序（详见 references/code_investigation.md）：

| 排查项 | 方法 | 优先级 |
|---|---|---|
| 失败用例直接调用的算子代码 | 对比 op_plugin 对应算子 opapi/aclops | P0 |
| 算子执行框架代码 | op_api_common.h EXEC_NPU_CMD 宏变更 | P0 |
| 内存分配器代码 | NPUCachingAllocator 逻辑变更 | P1 |
| deterministic/caching 机制 | OpParamMaker.cpp SetDeterministic | P1 |
| Copy/Cast kernel | CopyKernel*.cpp 变更 | P1 |
| Tensor factory | TensorFactories.cpp empty/empty_strided | P2 |
| Format cast | FormatCastKernelNpu.cpp | P2 |
| 测试文件本身 | 测试代码是否变更 | P2 |

### 阶段 6：下载并分析 CANN 日志（夜间模式默认跳过）

```bash
gh run download <RUN_ID> --repo Ascend/pytorch --name ascend-cann-logs-<SHARD> --dir cann_logs
mkdir -p cann_logs_ext && tar xzf cann_logs/ascend-cann-logs-<SHARD>.tar.gz -C cann_logs_ext
grep -ri "<算子名>" cann_logs_ext/ascend/log/debug/plog/ | head -20
grep -r "\[ERROR\]" cann_logs_ext/ascend/log/debug/plog/ | grep -i "<算子名>" | head
```

CANN 日志结构见 references/ci_structure.md。典型模式：
`CheckDtypeValid` 失败 → dtype 不支持（预期 fallback）；`GetWorkspaceSize` 失败 →
参数检查未通过；无 ERROR 但结果错误 → 输出未全部写入（内存未初始化等）。

## 根因分析框架

### 模式 1：数值精度问题

错误特征：`Tensor-likes are not close!`、`Mismatched elements`
1. 检查输出张量内存分配方式（是否初始化）
2. 检查 opapi 路径是否 fallback 到 aclop（DO_COMPATIBILITY）
3. 对比两条路径对边界场景（空输入/零维）的处理差异
4. 检查 CANN 算子是否写入全部输出

### 模式 2：算子执行错误

错误特征：`RuntimeError`、`aclnn xxx failed`
1. CANN plog ERROR 日志
2. dtype/格式支持性
3. fallback 是否正确

### 模式 3：Crash/超时

错误特征：`SIGSEGV`、`timeout`、`Segmentation fault`
1. 设备中毒（_check_npu_poisoned）
2. 内存越界
3. 多进程/多线程竞争

### 模式 4：回归分析（两流水线对比）

1. 直接代码变更：失败算子文件有修改 → 分析 diff
2. 框架层变更：op_api_common.h / OpParamMaker.cpp → 分析执行路径影响
3. 内存分配变更：NPUCachingAllocator → 分析输出内存初始化状态

## 产出格式

报告写入 `REPORT_PATH`（夜间模式）或直接输出（人工模式），结构：

```markdown
# AI 失败用例分析报告
## 1. 摘要（3-5 条结论，每条带置信度 高/中/低 与一句话依据）
## 2. 失败概览（按错误签名聚类：簇签名 × 用例数 × 代表 nodeid）
## 3. 基础设施类（不分析，一行清单/计数）
## 4. 用例根因分析（top N 逐条：现象 → 证据链 → 根因 → 置信度）
## 5. 回归判定（新回归/存量问题/偶现，附历史对比证据）
## 6. 建议动作（修复方向 / 白名单处理 / 重跑验证）
## 附：分析元信息（run id、commit、模型、用例覆盖数、是否下载 CANN 日志）
```

## 注意事项

1. **必须基于对应 commit 获取代码**：不能用本地/main 代码，代码可能已变化
2. **op_plugin 是子模块**：commit 独立于 Ascend/pytorch，需单独比对
3. **CANN 子模块可能不变**：两个 commit 间 runtime/ge 可能相同，问题在 torch_npu 层
4. **nightly 与 PR CI 测试集不同**：nightly 不做 skip_list 过滤
5. **PTA cache 是进程内缓存**：同 session 跨用例共享，哈希键变化影响后续算子
6. **CANN plog 可能不完整**：制品大小有限，只覆盖部分时间窗口
7. **~0.1s 通过的用例要警惕 vacuous pass**（上游裁剪/参数化空跑），对比总数
