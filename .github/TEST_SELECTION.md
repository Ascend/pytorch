# 测试选测机制说明：上游 Target Determination 与本仓精准选测

本文档说明两件事：

1. 上游 pytorch/pytorch 如何为每个 PR 决定"该跑哪些测试"（Target Determination，简称 TD）；
2. 本仓（Ascend/pytorch，torch_npu）PR 门禁的"精准选测"如何工作，与上游有何差异、有哪些已知盲区。

结论速览：

| 维度 | 上游 TD | 本仓精准选测 |
| --- | --- | --- |
| 决策形式 | 多信号打分，**跑分数最高的 25%**（排序式） | 覆盖率命中 + 白名单**交集**（过滤式） |
| 主要数据 | 历史失败/共现统计、LLM、路径关键词、PR 文本 | 覆盖率 release 里的 `test_case_map.json` |
| 粒度 | 测试文件 / 文件::测试类 | 文件（下游再按 nodeid 白名单收窄） |
| 结果缺失时 | 自动退化为跑全量 | 选测失败时回退全量；命中为空则跳过 |

---

## 第一部分：上游 Target Determination

### 1.1 定位

上游**没有**"精确选测"，也没有把覆盖率当作决策依据。它把"哪些测试该跑"建模为一个**打分问题**：
每个启发式（heuristic）对每个测试给出 `-1 ~ 1` 的置信分（负=建议跳过，0=无意见，正=建议运行，绝对值=置信度），
聚合后按分数排序，**只运行前 25%**，其余记录为 excluded。

### 1.2 四阶段流程

```
① 算分（独立 workflow）
   .github/workflows/target_determination.yml
     └─ 下载 pytest cache / LLM 产物 / 历史统计
     └─ uv run tools/testing/do_target_determination_for_s3.py
          └─ 载入 TESTS（tools/testing/discover_tests.py：test/ + test/cpp/）
          └─ 逐个跑启发式 → AggregatedHeuristics → td_results.json
          └─ 上报 metrics（最终优先级 + 每个启发式的明细）
     └─ 上传 artifact（S3 优先，fork PR 回退 GHA artifact）

② 分发
   各测试 job（_linux-test.yml/_mac-test.yml/_rocm-test.yml/_win-test.yml/_tpu-test.yml/_s390x-test.yml）
     └─ .github/actions/download-td-artifacts → .additional_ci_files/td_results.json

③ 应用（唯一决策点）
   .ci/pytorch/test.sh → python test/run_test.py ...
     └─ test/run_test.py 读 td_results.json，取 top 25% 作为 include，其余 excluded
     └─ 再按估算耗时切分片（tools/testing/test_selections.py）

④ 回流
   运行结果（失败用例、耗时）→ test-infra 统计（S3/ClickHouse）→ 成为下一轮启发式的输入
```

### 1.3 应用点：`test/run_test.py`

```python
selected_tests = get_selected_tests(options)        # 先按 --include/--exclude 等过滤
test_prioritizations = import_results()             # 读 .additional_ci_files/td_results.json
if len(test_prioritizations.get_all_tests()) == 0:
    options.enable_td = False                       # 结果缺失 → 跑全量
test_prioritizations.amend_tests(selected_tests)    # 对齐到本 job 的测试集合

percent_to_run = 25 if options.enable_td else 100
include, exclude = test_prioritizations.get_top_per_tests(percent_to_run)
test_batch = TestBatch("tests to run", include, False)
test_batch_exclude = TestBatch("excluded", exclude, True)
gen_ci_artifact([...include], [...exclude])          # CI 产物：实际跑/不跑清单
```

关键点：

- 阈值是**固定的 25%**，不是"命中才跑"；分数为 0 的测试也可能因为比例而进入 include；
- `TESTS_TO_INCLUDE` 环境变量存在时 `enable_td = False`（CI 里做定向验证用）；
- 分片按估算耗时切（`calculate_shards`），与 TD 解耦。

### 1.4 数据结构

| 类 | 作用 |
| --- | --- |
| `TestRun` | 运行单元：`file` 或 `file::Class`，支持 `&`/`\|`/`-` 集合运算，因此可表达"某文件里只跑几个类" |
| `TestPrioritizations` | 一个启发式对全部测试的分数表；`get_top_per_tests(n)` 取前 n%；`to_json/from_json` 即 `td_results.json` 格式 |
| `AggregatedHeuristics` | 收集各启发式结果；`get_aggregated_priorities()` 把分数**相加**（`trial_mode` 的不计入排序） |
| `HeuristicInterface` | 统一接口：`get_prediction_confidence(tests) -> TestPrioritizations`，并声明 `trial_mode` |

### 1.5 十个启发式

注册表见 `tools/testing/target_determination/heuristics/__init__.py`。

| 启发式 | 输入 | 逻辑 |
| --- | --- | --- |
| `PreviouslyFailedInPR` | previous_failures.json | 本 PR 之前失败的测试，+1（回归自愈） |
| `EditedByPR` | PR diff | PR 改动的测试文件 +1；`ADDITIONAL_MAPPINGS` 处理"跑别人测试的测试"，如 `test_cpp_extensions_aot_ninja` ↔ `test/cpp_extensions.*` |
| `MentionedInPR` | PR 正文 + commit 标题（含被引用的 issue/PR，深度 1） | 正则找测试名，+1 |
| `HistoricalClassFailurCorrelation` | test_class_ratings.json | 历史上"这些文件被改时容易失败的测试**类**"（`trial_mode=True`，只记 metrics） |
| `CorrelatedWithHistoricalFailures` | test_file_ratings.json | 历史"文件 ↔ 测试失败"相关性 |
| `HistorialEditedFiles` | historical_edited_files.json | 同一次 main 提交里被一起改的 **文件↔测试文件**（不需要覆盖率的历史共现） |
| `Profiling` | profiling.json | profiler 采集的 **Python 覆盖率**生成的 文件↔测试 关联（分全 1） |
| `LLM` | llm_results/mappings/*.json | CodeLlama + 索引器（`llm_td_retrieval.yml`）预测关联 |
| `Filepath` | PR diff + 关键词表 | 路径/文件名 → 关键词匹配测试关键词；**纯文档变更 → 全部 -1（跳过所有测试）** |
| `PublicBindings` | PR diff | 改动 `torch/**` → `test_public_bindings` +1（硬编码规则） |

### 1.6 `Filepath` 细节（与本仓最相关）

`tools/testing/target_determination/heuristics/filepath.py`：

- `get_keywords(file)`：把路径各级目录名 + 文件名按 `_` 切片成关键词；
- `keyword_synonyms`：同义词表，如 `amp→mixed_precision`、`quant→quantized/quantization`、`ops→opinfo`、`numpy→torch_np`、`inductor→dynamo/export`；
- `custom_matchers`：特殊匹配，如 `nn`（把 `onnx` 里的 nn 排除）、`c10`（排除 `c10d`）；
- `is_valid_keyword`：过滤 `torch/test/util/src/c/tools/internal` 等噪声词；
- `get_freq_dict`：按关键词出现频率给测试打分，再 `normalize_ratings(..., 0.25, min_value=0.125)`。

**C/C++ 变更就是靠这一条参与映射的**——是关键词近似，不是精确的"哪个 C 文件被哪个测试覆盖"。

### 1.7 数据来源与回流

- 统计由 **test-infra 侧**的聚合任务生成（test times、test/class ratings、historical edited files、profiling、previous failures），CI 里只是下载；
- 每次跑都通过 `emit_metric` 上报：最终优先级 JSON、各启发式 JSON；
- 失败用例与耗时回流，供下一轮启发式使用；
- `trial_mode` 支持灰度：新启发式先只产出 metrics，不影响实际排序。

### 1.8 容错与旁路

| 情形 | 行为 |
| --- | --- |
| `td_results.json` 不存在 / 为空 | `enable_td = False` → 跑全量 |
| `TESTS_TO_INCLUDE` 指定 | 关闭 TD，按指定集合跑 |
| LLM 检索失败 | 该 workflow `continue-on-error`，`LLM` 启发式返回空映射 |
| 仅文档变更（.rst/.md） | `Filepath` 返回全 -1，等价于跳过所有测试 |
| 下载 TD artifact 失败 | job 侧 `continue-on-error`，退化为跑全量 |

---

## 第二部分：本仓精准选测

### 2.1 触发入口与闸门

```
上游 pytorch/pytorch webhook
  └─ CRCR relay（无状态透传，转发完整 payload）
       ├─ repository_dispatch: pull_request → pytorch_ci_trigger_pr.yml / pytorch_ci_trigger_precise.yml
       └─ repository_dispatch: push         → pytorch_ci_trigger_push.yml
            └─ 均转调 receive-trigger.yml（ci-test）
                 ├─ extract：闸门
                 │    ├─ pull_request：必须 action=closed 且带 Merged label（缺失时回退实时 API 查询）
                 │    └─ push：必须是 pytorch-bot 创建的 refs/tags/ciflow/trunk/<PR#>
                 ├─ build  （should_build=true 才有）
                 ├─ select （仅 pull_request → _precise_test.yml）
                 └─ test   （_test.yml → _test-collect.yml → _test-category.yml）
```

不满足闸门的事件会在 `extract` 直接 `exit 0`（`should_build=false`），所以 run 里只有 extract 一个 job、总时长数秒。

### 2.2 覆盖率数据的产生（`_pytorch_coverage.yml`）

- 采集范围：`nightly_test_categories.yml`（全量扫描分类）+ `full_scan: true` + `privateuse1_npu_passed_testcases.json`（用例级白名单）；
- 逐用例数据：pytest 插件 `zz_cov_plugin`（`coverage.Coverage`，`source = torch`）→ 每个用例一份 `covdata`，按测试文件分组；
- merge job：合并分片 → `coverage.xml` + `manifest.json`，并用同一 action 的 `build-map: true` 生成 `test_case_map.json`（测试 → 覆盖的源码文件+行）；缺失/失败文件用上一版 release 回填；
- 发布：打包 `covstub`（源码快照）+ `pytorch@latest` → `obs://pytorch-ci-cache/coverage/pytorch/latest/release.tar.gz`（带内容校验与 `obsutil stat` 校验）；
- 触发：每日 cron + `pytorch_coverage_nightly.yml` 的 push paths（含白名单文件）。

> 注意：覆盖率**只覆盖 Python 的 `torch/` 包**，C/C++ 部分没有任何覆盖率数据（详见第三部分盲区）。

### 2.3 选测器规则（`_precise_test.yml`）

- 下载并解包 OBS release，得到 `pytorch@latest/`（含 `test_case_map.json`）与 `covstub/`；
- 调 `lb-actions/precision-test@v1.0.4`（`repo: pytorch`、`github-pr`、`source-dir`、`coverage-dir`、`map-file`、`dedup: true`），产出 `recommended_pytest_paths.txt`；
- 选测器两条来源（并集去重）：
  1. **覆盖率映射命中**：PR 改动的源码文件/行 → 在 `test_case_map.json` 中找覆盖这些行的测试 → 取其测试文件（行级 → 函数级 → 文件级级联，文件级仅用于重命名/删除的产品代码）；
  2. **PR 改动的测试文件直加**：diff 中新增/修改的 `test/**/test_*.py` 直接进推荐，**不查覆盖率**；
- 另有：删除的测试文件从结果移除；命中"全量触发变更"时推荐全部测试。

### 2.4 白名单过滤与用例级收窄

- **文件级**（`_precise_test.yml`）：从 ci-test 取 `privateuse1_npu_passed_testcases.json`，用 `nodeid.split("::")[0]` 得到文件集，与推荐清单取交集，打印 kept/dropped；
- **用例级**（`_test-collect.yml`）：生成 `precise_case_whitelist.json` = 调用方白名单中"所属文件在推荐清单内"的 nodeid，作为 `--case-whitelist` 传给收集脚本。

两级闸门**同源**（都是 PR 门禁本来会跑的那批文件/用例），保证精准跑是"全量的子集"。

### 2.5 精准模式的分片划分与 runner 路由（`_test-collect.yml`）

- `precise_files` 非空时：
  - 生成 `precise_cases.yml`：复用全量同一套分类实现（`load_categories_config` + `classify_files_full_scan`，files > paths > others 三段式），即"全量怎么分，精准就怎么分"；
  - 强制不启用 `--full-scan`（避免又去全量扫描）；
  - runner 路由：`distributed` 用调用方传入的 `runner_label`（receive-trigger 传 `linux-aarch64-a3-800i-2-cn12-001`），其余分类（core/tensor/graph/others）硬编码 `linux-aarch64-a3-800i-8-cn12-001`（32-worker 批量分片需要 8 卡机内存）；
  - 阈值沿用调用方参数（regular 50000 / distributed 3000）。

### 2.6 判定矩阵

| 情况 | 结果 |
| --- | --- |
| prepare 判定非落地 PR / 非 pull_request | 选测跳过，`precise_files` 为空，整体成功 |
| 选测器执行失败 | `test` 检测到 `select.result != success` → **回退全量** |
| 推荐 ∩ 白名单 = 空集 | **跳过测试阶段**（仅 PR 事件） |
| 推荐 ∩ 白名单 ≠ 空 | 只跑这些文件，内部再按 nodeid 白名单收窄 |
| push（ciflow/trunk） | 不走选测，全量（白名单全集） |
| manual（workflow_dispatch / push 触发） | 不走选测，全量（nightly 分类全扫描 ∩ 白名单） |

---

## 第三部分：已知盲区与风险

1. **C/C++ 代码没有任何覆盖率数据**（这是根本限制）：
   - 采集端 `.coveragerc` 为 `source = torch`，逐用例数据来自 Python 的 `coverage.Coverage`；
   - 构建端没有任何插桩（无 `-fprofile-instr-generate`/`--coverage`），wheel 是普通 release 构建；
   - 合并端 `[paths] torch = covstub/torch + site-packages/torch`，源码快照只含 `pytorch/torch` 与 `pytorch/test`，**连 `torch/csrc`、`aten/` 的源码都不在快照里**；
   - **后果**：PR 若只改 C/C++，行级/函数级匹配 0 命中、文件级匹配也不适用（仅用于重命名/删除）→ 推荐为空 → **跳过测试阶段**。

2. **`repo: pytorch` 适配器与上游路径不完全对齐**：v1.0.x 的适配器注册表中 `pytorch` 映射到 `torch_npu` 适配器（产品前缀 `torch_npu/`），而上游覆盖率路径是 `.../site-packages/torch/...`，导致剥前缀失败、覆盖率映射这一路基本失效；实际推荐主要来自"PR 改动测试文件直加"。

3. **PR 落地事件的标签时序竞态**：MergeBot 可能先 `closed` 再补 `Merged` 标签；若 closed 事件到达时 payload 与实时 API 都还没有该标签，extract 会判 `SKIP`，而随后补标签产生的 `labeled` 事件会被 `action != 'closed'` 挡掉，**该 PR 不会再有触发机会**。

4. **白名单是活配置、覆盖率数据是快照**：文件级/用例级闸门读 ci-test HEAD 的白名单，覆盖率 release 是上一次跑出来的数据，窗口内两者可能不一致（新进白名单的文件暂时没有覆盖率映射）。

---

## 第四部分：可借鉴项

1. **`Filepath` 关键词兜底（推荐先做）**：对 C/C++ 变更（`torch/csrc/**`、`aten/**`、`c10/**`、`*.cpp/*.h/*.cu`）按上游关键词法生成候选测试文件加入推荐；完全匹配不到时**退化全量**。零插桩，能立刻堵住"C++ PR 被跳过测试"的洞。
2. **多信号打分取代硬交集**：单一来源失效时，打分（负/0/正）比"空集即跳过"稳健；可先用 `trial_mode` 式灰度只记录、不影响决策。
3. **度量闭环**：把每次的 include/exclude 清单与各信号排名上报，用于评估漏跑（这正是判断兜底是否有效的唯一手段）。
4. **容错范式**：决策数据缺失（release 不存在 / map 为空 / 选测失败）一律退化全量，绝不以"空集"为结论。
5. **粒度**：上游支持 `file::Class`，本仓目前只到文件 + nodeid；若要更细可参考 `TestRun` 的集合运算设计。

---

## 第五部分：关键文件索引

上游（pytorch/pytorch）：

| 文件 | 说明 |
| --- | --- |
| `.github/workflows/target_determination.yml` | TD 计算 workflow（算分 + 上传 artifact） |
| `tools/testing/do_target_determination_for_s3.py` | TD 入口：跑启发式、聚合、写 `td_results.json`、上报 metrics |
| `tools/testing/target_determination/determinator.py` | 启发式调度 |
| `tools/testing/target_determination/heuristics/__init__.py` | 启发式注册表 |
| `tools/testing/target_determination/heuristics/interface.py` | `TestPrioritizations` / `AggregatedHeuristics` / 接口定义 |
| `tools/testing/target_determination/heuristics/filepath.py` | 路径关键词启发式 |
| `tools/testing/target_determination/heuristics/utils.py` | 变更文件查询、评分归一化等公共工具 |
| `tools/testing/test_run.py` | `TestRun`（file / file::Class 粒度） |
| `tools/testing/test_selections.py` | 分片计算（按耗时） |
| `tools/testing/discover_tests.py` | 测试发现（含 `test/cpp/**`，加 `cpp/` 前缀） |
| `test/run_test.py` | **应用点**：读 TD 结果、取 top 25%、分片、上报 |
| `.ci/pytorch/test.sh` | CI 调用入口 |

本仓（Ascend/pytorch，ci-test 分支）：

| 文件 | 说明 |
| --- | --- |
| `.github/workflows/receive-trigger.yml` | 上游事件入口 + extract 闸门 + 全量/精准的分发 |
| `.github/workflows/pytorch_ci_trigger_pr.yml` / `_push.yml` / `_precise.yml` / `_manual.yml` | 各类 repository_dispatch / 手动入口 |
| `.github/workflows/_precise_test.yml` | 精准选测（OBS 下载、selector、白名单文件级过滤、输出 `precise_files`） |
| `.github/workflows/_test.yml` | 测试总编排（collect → 分片 → 报告） |
| `.github/workflows/_test-collect.yml` | 用例收集/分片；精准模式生成 `precise_cases.yml` 与 `precise_case_whitelist.json` |
| `.github/workflows/_test-category.yml` | 单分类分片执行 |
| `.github/workflows/_pytorch_coverage.yml` + `_pytorch_coverage_shard.yml` | 覆盖率采集、合并、`test_case_map.json` 生成、发布 OBS（用例收集复用 `_test-collect.yml`） |
| `.github/workflows/pytorch_coverage_nightly.yml` | 覆盖率触发（每日 cron + push paths） |
| `.github/config/nightly_test_categories.yml` | 全量分类配置（分类边界 + 每类 workers/execution/runner） |
| `.github/config/privateuse1_npu_passed_testcases.json` | 用例白名单（passed nodeids，当前约 6.7 万条 / 58 个文件） |
| `.github/scripts/collect_all_cases.py` | 用例收集/分类/白名单过滤/分片实现 |
| `.github/scripts/run_pytorch_coverage.sh` | 覆盖率采集与合并脚本（逐用例 covdata 归集） |
