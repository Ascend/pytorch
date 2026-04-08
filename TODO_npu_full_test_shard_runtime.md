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

## 建议的落地设计稿（混合加权分片）

基于当前观察，建议优先落地“普通文件继续按文件分片，超重文件按更细粒度拆分，再用历史耗时做加权均衡”的混合策略。

### 设计目标

- 尽量降低长尾 shard 的耗时，而不是追求理论最优调度
- 尽量少改动现有 workflow 和 runner，优先复用当前执行链路
- 保持分片结果相对稳定，避免每次运行都大幅波动
- 当没有历史数据时也能正常退化运行
- 不在本设计中改变 pytest 的失败策略，例如暂不引入 `-x` 或 `--maxfail`

### 核心思路

引入“测试执行单元（Test Unit）”的概念，不再把所有内容都简单视为“1 个测试文件”。

初版规则如下：

1. 普通测试文件仍然作为 1 个执行单元
2. 少数已知超重文件会被拆成多个执行单元
3. 分片时按“执行单元”的预估权重做负载均衡，而不是按文件数均分
4. 权重优先使用历史耗时，缺失时再退化到 collect 数量或默认值

这样可以兼顾：

- 绝大多数文件保持当前行为
- 只对少数极重文件做定向治理
- 后续可以平滑演进到更细粒度的历史耗时调度

### 执行单元抽象

建议在 prepare 阶段生成一个 manifest，把要执行的内容统一描述成执行单元。

每个执行单元建议至少包含以下字段：

- `id`：全局唯一、稳定的单元标识
- `source_file`：来源测试文件
- `kind`：执行类型，例如 `file`、`selector`、`selector_with_env`
- `selectors`：pytest 目标列表，例如文件路径、`file::Class`、`file::Class::test_xxx`
- `env`：该单元特有的环境变量覆盖
- `estimated_cases`：预估 testcase 数量
- `estimated_duration`：预估耗时
- `stable_key`：用于稳定排序的键

建议 manifest 结构类似：

```json
{
  "schema_version": 1,
  "units": [
    {
      "id": "inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU::range[0,343)",
      "source_file": "inductor/test_torchinductor_opinfo.py",
      "kind": "selector_with_env",
      "selectors": [
        "inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU"
      ],
      "env": {
        "PYTORCH_TEST_RANGE_START": "0",
        "PYTORCH_TEST_RANGE_END": "343"
      },
      "estimated_cases": 1800,
      "estimated_duration": 420.0,
      "stable_key": "inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCPU::0000-0343"
    }
  ],
  "shards": {
    "58": [
      "unit_id_1",
      "unit_id_2"
    ]
  }
}
```

### 初版超重文件白名单

根据当前 shard 58 的实际结果，初版建议只对白名单中的超重文件做特殊拆分：

- `inductor/test_torchinductor_opinfo.py`
- `inductor/test_torchinductor_dynamic_shapes.py`
- `inductor/test_torchinductor_codegen_dynamic_shapes.py`
- `inductor/test_torchinductor.py`

其余文件先继续保持按文件执行：

- `inductor/test_torchinductor_strided_blocks.py`
- `inductor/test_torchinductor_codegen_config_overrides.py`
- `inductor/test_triton_cpu_backend.py`
- `inductor/test_triton_extension_backend.py`

这样可以先解决最主要的长尾来源，避免一开始就把调度逻辑做得过重。

### 初版拆分规则

#### 1. `inductor/test_torchinductor_opinfo.py`

这是首要治理对象，建议优先特殊拆分。

拆分方式建议如下：

1. 先按测试类拆分
2. 再在每个测试类内部按 `op_db` 范围拆分

初版建议至少拆成以下 4 个单元：

- `TestInductorOpInfoCPU` 的前半段
- `TestInductorOpInfoCPU` 的后半段
- `TestInductorOpInfoPRIVATEUSE1` 的前半段
- `TestInductorOpInfoPRIVATEUSE1` 的后半段

如果后续仍然偏重，可以继续扩展到 8 个单元。

之所以优先使用这种拆分方式，是因为该文件源码已经支持通过环境变量切 `op_db` 范围：

- `PYTORCH_TEST_RANGE_START`
- `PYTORCH_TEST_RANGE_END`

因此不需要一开始就改成完全按 nodeid 级别拆分。

#### 2. `inductor/test_torchinductor_dynamic_shapes.py`

该文件建议先按类拆分：

- `DynamicShapesCpuTests`
- `TestInductorDynamicCPU`
- `TestInductorDynamicPRIVATEUSE1`

其中：

- `DynamicShapesCpuTests` 是主要负载来源，权重应视为重单元
- `TestInductorDynamicCPU`
- `TestInductorDynamicPRIVATEUSE1`

后两者在当前环境下多数情况下会直接 skip，因此历史权重通常应很低，不能再和主类混在一起估重。

#### 3. `inductor/test_torchinductor_codegen_dynamic_shapes.py`

该文件建议至少拆成 2 个单元。

由于它主要由单个大类 `DynamicShapesCodegenCpuTests` 展开而来，初版可以采用以下方式之一：

- 按 collect 到的 nodeid 均分为 2 份
- 按稳定哈希把 nodeid 分到 2 个桶

初版更建议使用“collect 后均分 nodeid”的方式，因为更直观，也更容易和实际 case 数对应。

#### 4. `inductor/test_torchinductor.py`

该文件建议先按类拆分：

- `CpuTests`
- `SweepInputsCpuTest`
- `TestFull`

其中：

- `CpuTests` 是主要负载来源
- `SweepInputsCpuTest` 只有自动生成的输入布局组合，规模中等
- `TestFull` 只有极少量用例

这种拆分粒度足以显著减少该文件单独成为长尾的概率。

### 权重来源与估算规则

建议按以下优先级为每个执行单元估重：

1. 最近若干次运行的历史中位耗时
2. 历史中位 collect 数量
3. 默认权重 `1`

推荐的具体规则如下：

- 若某执行单元已有至少 3 次有效历史数据，则权重使用最近 3 到 5 次的中位耗时
- 若没有稳定耗时数据，但有 collect 数量，则按 collect 数量乘固定系数换算权重
- 若某执行单元连续多次 `0 collected`，则权重记为 `0`
- 若某执行单元长期高 skip 比例，则优先相信历史耗时，不要简单按 collect 数量估重

这里要特别注意：

- `case 数量` 只适合作为冷启动估算
- 真正用于均衡时，`历史耗时` 更可信
- 因为当前很多 inductor 类测试的真实成本并不只来自 testcase 自身，还来自 collection、compile、日志和报告链路

### 超重文件判定阈值

为了避免白名单无限膨胀，建议增加一个简单的进入阈值。

执行单元或文件满足以下任一条件时，可进入“特殊拆分候选名单”：

- 最近多次运行中位耗时超过 `300s`
- collect 到的 testcase 数量超过 `1000`
- 在最慢 shard 中连续多次出现

相反，如果某个已拆分文件连续多次满足以下条件，可以考虑移出白名单，恢复普通按文件执行：

- 中位耗时低于 `120s`
- collect 数量较低
- 不再频繁出现在长尾 shard 中

### 分片算法

prepare 阶段建议采用简单、稳定、可解释的贪心装箱策略。

建议步骤如下：

1. 发现所有测试文件
2. 对普通文件生成 1 个 `file` 单元
3. 对白名单重文件生成多个更细粒度的执行单元
4. 读取历史权重并为每个单元估重
5. 按 `estimated_duration` 降序、`stable_key` 升序排序
6. 依次把当前最重单元放入“当前总权重最小”的 shard

这种策略的优点是：

- 实现简单
- 行为稳定
- 不依赖复杂优化器
- 对“少数超重单元 + 大量普通单元”的场景很适合

### runner 改造建议

当前 `.github/scripts/run_npu_test_shard.py` 的输入是“一个 shard 对应的一组测试文件”。建议把它扩展为“一个 shard 对应的一组执行单元”。

初版建议新增以下能力：

- 支持从 manifest 读取当前 shard 的执行单元列表
- 支持执行文件级 selector
- 支持执行类级 selector
- 支持执行带环境变量覆盖的 selector
- 支持一个 shard 内多次调用 pytest，并把结果合并

为了控制改动范围，建议分两层执行：

1. 普通文件单元仍尽量合并为 1 次 pytest 调用
2. 特殊拆分单元按单元逐个执行，或按相同环境变量分组执行

这样可以避免：

- 所有 shard 都因为多次 pytest 启动而增加额外开销
- 简单文件也被迫走复杂路径

### JUnit 与统计文件处理

由于一个 shard 可能不再只有 1 次 pytest 调用，建议 runner 输出分段报告，再做聚合：

- `junit_shard_<n>_part_<k>.xml`
- `test_shard_<n>_part_<k>.log`
- `shard_<n>_unit_stats.json`

最终再额外生成一个聚合后的：

- `junit_shard_<n>.xml`
- `shard_<n>_stats.json`
- `shard_<n>_planned_test_units.txt`

聚合时建议同时保留每个执行单元的以下信息：

- collect 数量
- passed / failed / skipped
- wall clock duration
- JUnit testcase time 求和
- selector 列表
- 环境变量覆盖

这样后续做历史耗时加权时就有稳定的数据来源。

### prepare / report 阶段改造建议

建议新增一个 prepare 阶段的 manifest 生成步骤，例如：

- 扫描测试文件
- 识别白名单重文件
- 生成执行单元
- 基于历史数据完成分片
- 把 manifest 作为 artifact 传给各个 test job

report 阶段建议新增一个聚合步骤：

- 汇总所有 shard 的 `unit_stats`
- 计算每个执行单元的最新耗时和 collect 数量
- 产出下一次运行可复用的历史权重文件

历史权重文件可以先保存在 workflow artifact 中，后续如果需要再演进到仓库内维护或外部持久化。

### 建议新增的中间产物

为了便于调试和回溯，建议初版新增以下文件：

- `test-reports/test_sharding_manifest.json`
- `test-reports/shard_<n>_planned_test_units.txt`
- `test-reports/shard_<n>_unit_stats.json`
- `test-reports/test_unit_history.json`

这样排查时可以直接回答以下问题：

- 某个 shard 为什么会很重
- 某个重文件被拆成了哪些单元
- 某个单元的历史权重从哪里来
- 某个单元这次是实际执行了、全部 skip 了，还是根本没有 collect 到

### 分阶段落地建议

建议按以下顺序渐进实施，而不是一次性重构全部逻辑。

#### 阶段 1：最小可用版本

- 保持普通文件仍按文件执行
- 只对白名单中的 `test_torchinductor_opinfo.py` 做特殊拆分
- 使用简单的历史文件级耗时作为权重
- runner 支持单 shard 多次 pytest 调用和结果聚合

目标是先解决最明显的单点长尾。

#### 阶段 2：扩展到另外 3 个重文件

- 加入 `test_torchinductor_dynamic_shapes.py`
- 加入 `test_torchinductor_codegen_dynamic_shapes.py`
- 加入 `test_torchinductor.py`
- 权重粒度从“文件级”提升到“执行单元级”

目标是把当前已知的主要长尾来源都纳入治理。

#### 阶段 3：补齐冷启动与自动化收敛

- 为新出现的重文件增加自动判定阈值
- 对缺失历史数据的单元使用 collect 数量兜底
- 根据多次运行结果自动调整白名单和拆分份数

到这个阶段，再决定是否值得进一步演进到更通用的“全量按历史耗时调度”。

### 成功判定标准

后续如果正式实现，建议至少关注以下指标：

- 最慢 shard 耗时是否显著下降
- shard 耗时的离散度是否明显收敛
- 中位 shard 耗时是否没有因为额外 pytest 启动而明显恶化
- 最慢 shard 是否不再总是集中在少数几个 inductor/opinfo 文件上

如果这些指标明显改善，就说明混合加权分片策略是有效的。

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
