# NPU 全量测试分片耗时优化待办

状态：暂缓，后续有时间时再优化

## 适用范围

本文档记录当前 GitHub Actions 全量测试流程中的一个已知问题，涉及以下文件：

- `.github/workflows/npu-full-test.yml`
- `.github/scripts/run_npu_test_shard.py`

当前先不实施优化，只将问题背景、原因和后续优化思路记录下来，方便后续继续推进。

## 问题背景

当前分片逻辑是按测试文件进行分片，而不是按 pytest 实际收集到的 testcase 数量分片，也不是按历史执行耗时分片。

当前逻辑大致是：

1. 发现所有 `test_*.py` 文件
2. 对文件路径排序
3. 按文件数平均切分为多个 shard
4. 每个 shard 启动一个 pytest 进程执行属于该 shard 的测试文件

这种方式的优点是：

- 逻辑简单
- 分片稳定
- 在测试文件集合不变时，同一文件会稳定落在固定 shard 中

但它有一个明显问题：

- “文件数均衡”不代表“执行耗时均衡”

尤其是某些测试文件虽然只有 1 个文件，但由于参数化非常重，实际会展开成成千上万条 pytest 用例，导致某些 shard 明显比其他 shard 慢很多。

## 已观察到的典型案例

参考任务：

- Workflow run: https://github.com/Ascend/pytorch/actions/runs/24113337524
- Job: https://github.com/Ascend/pytorch/actions/runs/24113337524/job/70352457274
- 日期：2026 年 4 月 8 日
- Job 名称：`test (58)`
- 分片：`58/100`

该任务中：

- `Run shard 58/100` 开始时间：`2026-04-08T02:01:20Z`
- `Run shard 58/100` 结束时间：`2026-04-08T03:25:15Z`
- 单步耗时约：`1 小时 23 分 55 秒`

该 shard 的统计结果为：

- `total = 9829`
- `passed = 200`
- `failed = 8329`
- `skipped = 1300`
- `errors = 0`
- `duration ≈ 4960.6s`

从表面上看，这个 shard 只分到少量测试文件，但实际展开后产生了将近一万条 pytest testcase。

## 主要原因分析

### 1. 当前按“测试文件数”分片，无法反映真实执行规模

当前分片逻辑把每个测试文件视为同等权重：

- 一个只包含少量 testcase 的文件
- 一个会展开出数千条参数化用例的文件

在分片阶段都只算“1 个文件”

但在实际执行阶段，两者的耗时可能相差几十倍甚至上百倍。

### 2. 某些 inductor / opinfo 类测试文件参数化非常重

根据本次 `shard 58` 的 JUnit 结果聚合，主要耗时和用例数量集中在以下模块：

- `test.inductor.test_torchinductor_opinfo`
- `test.inductor.test_torchinductor_dynamic_shapes`
- `test.inductor.test_torchinductor`
- `test.inductor.test_torchinductor_codegen_dynamic_shapes`

其中最大头是：

- `test.inductor.test_torchinductor_opinfo`
- 单个模块就展开出约 `7214` 条 testcase

这说明：

- 这个 shard 看起来只分到了少量文件
- 但 pytest 实际运行时负载极重

### 3. 当前没有早停机制

当前 workflow 没有使用：

- `-x`
- `--maxfail`

因此即使前面已经失败了很多 testcase，pytest 仍会继续执行整个 shard 中剩余测试。

这会进一步放大慢 shard 的总耗时。

### 4. 大量失败还会放大日志和报告开销

当一个 shard 中有几千个失败用例时，除了测试本身耗时，还会额外增加：

- stdout 输出量
- JUnit XML 写入量
- 日志 I/O
- GitHub Actions 页面渲染和 artifact 体积

因此单条失败显示只要几十毫秒，并不代表整个 shard 很轻。

### 5. 单条 testcase 时间不等于整个 shard 的真实耗时

从本次数据看：

- pytest 打印总耗时约 `5014.44s`
- stats 记录总耗时约 `4960.62s`
- 但把 JUnit XML 里每条 testcase 的 `time` 简单求和，远小于这个数字

说明实际大量时间花在以下环节：

- collection
- setup / teardown
- 编译 / 动态形状准备
- 子进程或底层框架初始化
- 大量失败后的报告输出

因此当前的耗时问题并不只是“某些 case 自身执行慢”，而是“整个测试文件族在 pytest 执行链路上都很重”。

## 问题影响

该问题会带来以下影响：

- shard 之间负载严重不均
- 某些 shard 成为长尾任务，拖慢整个 workflow 完成时间
- runner 利用率不均衡
- 大量失败日志会进一步放大总耗时
- 分片数虽然很多，但并不能保证整体并行效率高

## 后续可考虑的优化方向

以下是后续可以考虑的优化思路，目前暂不实施。

### 方案 1：按 pytest collect 后的 testcase 数量分片

思路：

- 先做一次 `pytest --collect-only`
- 统计每个测试文件最终展开出的 testcase 数量
- 分片时不再按“文件数”均分，而按“预计 testcase 数量”均分

优点：

- 比单纯按文件分片更合理
- 仍然可以保持相对稳定

缺点：

- 会增加一次 collect 开销
- testcase 数量仍然不完全等于真实耗时

### 方案 2：按历史执行耗时分片

思路：

- 记录每个测试文件或测试模块的历史耗时
- 后续分片时按预计总耗时做负载均衡

优点：

- 理论上最接近真实负载均衡

缺点：

- 需要维护历史数据
- 代码变更或环境变化后，历史耗时可能漂移

### 方案 3：对超重文件做特殊拆分

思路：

- 对已知的超重测试文件单独特殊处理
- 例如把 `test_torchinductor_opinfo.py` 这类文件再按 nodeid、测试类、设备类型、dtype 等维度二次切分

优点：

- 针对性强
- 改动范围相对可控
- 适合先解决少数最重文件

缺点：

- 需要维护特殊规则
- 规则可能随 upstream 变化而调整

### 方案 4：混合策略

思路：

- 大多数普通文件仍沿用当前按文件分片
- 仅对少数已知超重文件做特殊处理

优点：

- 工程成本低于完整重做调度逻辑
- 往往足以解决最明显的长尾问题

缺点：

- 不是全局最优
- 仍然需要维护部分特殊名单

## 建议的后续推进顺序

如果后续有时间推进，建议优先采用“混合策略”：

1. 先统计最近几次 workflow 中最慢的 shard
2. 找出其中最重的测试文件
3. 对极少数明显超重的文件做特殊拆分
4. 如果效果仍不理想，再考虑演进到“按历史耗时分片”

这样实施成本和收益更平衡。

## 后续优化前建议先补充采集的数据

后续如果要正式优化，建议先收集以下信息：

- 最近多次 workflow 中最慢的前若干个 shard
- 每个 shard 的 testcase 总数
- 每个测试文件展开后的 testcase 数量
- 每个测试文件的总耗时贡献
- 是否确实是少数几个 inductor/opinfo 文件导致大多数长尾 shard

## 当前结论

当前问题已经确认存在：

- 当前按文件分片的方式，在超重参数化测试文件场景下会导致 shard 耗时严重不均

但本轮暂不实施优化。

后续如需继续推进，可基于本文档继续分析和设计实现方案。
