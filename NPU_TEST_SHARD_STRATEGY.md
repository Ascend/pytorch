# PyTorch NPU 测试分片策略文档

本文档详细说明 PyTorch NPU 测试的分片策略、执行逻辑以及 whitelist/blacklist 配置机制。

---

## 一、分片架构

### 1.1 分片分配

| Shard | 类型 | 测试范围 | 机器数 |
|-------|------|----------|--------|
| 1-2 | distributed | `test/distributed/*` | 2 |
| 3 | excluded | 被 discover_tests.py 排除的测试 | 1 |
| 4-10 | regular | 其他测试 | 7 |

**总计**: 10 个 Shard

### 1.2 分片常量定义

```python
# 分布式测试分片
SHARD_DISTRIBUTED_START = 1
SHARD_DISTRIBUTED_END = 2
SHARD_DISTRIBUTED_TOTAL = 2

# 排除测试分片（blocklisted patterns/tests）
SHARD_EXCLUDED_START = 3
SHARD_EXCLUDED_END = 3
SHARD_EXCLUDED_TOTAL = 1

# 常规测试分片
SHARD_REGULAR_START = 4
SHARD_REGULAR_END = 10
SHARD_REGULAR_TOTAL = 7

TOTAL_SHARDS = 10
```

---

## 二、测试发现机制

### 2.1 各 Shard 的测试发现方式

| Shard 类型 | 发现方式 | 说明 |
|------------|----------|------|
| distributed | TESTS 列表 (`discover_tests.py`) | 与 `run_test.py` 一致 |
| regular | TESTS 列表 (`discover_tests.py`) | 与 `run_test.py` 一致 |
| excluded | **原始文件扫描** | 需要找到被 `discover_tests.py` 排除的测试 |

### 2.2 为什么 Excluded Shard 使用原始扫描

`discover_tests.py` 通过 `blocklisted_patterns` 和 `blocklisted_tests` 排除了以下测试：

**blocklisted_patterns (目录排除)**:
- `test/custom_backend`
- `test/custom_operator`
- `test/fx`
- `test/mobile`
- `test/quantization`

**blocklisted_tests (文件排除)**:
- `test/test_bundled_images.py`
- `test/test_cpp_extensions_aot.py`
- `test/test_determination.py`
- `test/test_jit_string.py`
- `test/test_kernel_launch_checks.py`
- `test/test_nnapi.py`
- `test/test_static_runtime.py`
- `test/test_throughput_benchmark.py`

这些测试不在 TESTS 列表中，如果使用 TESTS 列表发现，Shard 3 将找不到任何测试。因此必须使用**原始文件扫描**来发现这些被排除的测试。

---

## 三、Excluded Shard 测试范围

### 3.1 EXCLUDED_TESTS_PATTERNS

```python
EXCLUDED_TESTS_PATTERNS = [
    # blocklisted_patterns from discover_tests.py (directories)
    "test/custom_backend",      # all test files in this directory
    "test/custom_operator",     # all test files in this directory
    "test/fx",                  # all test files in this directory
    "test/mobile",              # all test files in this directory
    "test/quantization",        # all test files in this directory
    # blocklisted_tests from discover_tests.py (individual files)
    "test/test_bundled_images.py",
    "test/test_cpp_extensions_aot.py",
    "test/test_determination.py",
    "test/test_jit_string.py",
    "test/test_kernel_launch_checks.py",
    "test/test_nnapi.py",
    "test/test_static_runtime.py",
    "test/test_throughput_benchmark.py",
]
```

### 3.2 动态匹配函数

```python
def path_matches_excluded_pattern(path: str) -> bool:
    """
    Check if a test file path matches any excluded test pattern.
    """
    for pattern in EXCLUDED_TESTS_PATTERNS:
        if pattern.endswith('.py'):
            # Exact file match
            if path == pattern:
                return True
        else:
            # Directory match: path starts with pattern + '/'
            if path.startswith(pattern + '/'):
                return True
    return False
```

### 3.3 匹配结果统计

| 目录 | 匹配文件数 |
|------|-----------|
| test/custom_backend | 1 |
| test/custom_operator | 2 |
| test/fx | 21 |
| test/mobile | 6 |
| test/quantization | 47 |
| blocklisted_tests (文件) | 8 |
| **总计** | **85** |

---

## 四、完整执行流程

### 4.1 Distributed Shard (1-2)

```
┌─────────────────────────────────────────────────────────┐
│ Shard 1-2: Distributed Tests                            │
├─────────────────────────────────────────────────────────┤
│ 1. get_tests_list_from_discover_tests()                 │
│    → TESTS 列表 (discover_tests.py)                     │
│                                                          │
│ 2. filter_tests_by_type("distributed")                  │
│    → 只保留 test/distributed/*                          │
│                                                          │
│ 3. apply_case_path_rules(whitelist, blacklist)          │
│    → whitelist: test/distributed 在 whitelist 中        │
│    → blacklist: 排除 17 个不支持测试                     │
│                                                          │
│ 4. select_shard_files(shard_index, shard_total)         │
│    → 按文件数量均分到 Shard 1 和 Shard 2                │
│                                                          │
│ 5. run_tests_via_run_test()                             │
│    → 使用 run_test.py 执行                              │
│    → NUM_PARALLEL_PROCS=4 (分布式测试降低并发)          │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Excluded Shard (3)

```
┌─────────────────────────────────────────────────────────┐
│ Shard 3: Excluded Tests                                 │
├─────────────────────────────────────────────────────────┤
│ 1. discover_raw_test_files()                            │
│    → 原始文件扫描 (所有 test_*.py)                      │
│                                                          │
│ 2. filter_tests_by_type("excluded")                     │
│    → 匹配 EXCLUDED_TESTS_PATTERNS                       │
│    → 结果: 85 个文件                                    │
│                                                          │
│ 3. apply_case_path_rules(whitelist, blacklist)          │
│    → whitelist: 85 个文件全部在 whitelist 中            │
│    → blacklist: 排除 6 个测试                           │
│    → 结果: 79 个文件                                    │
│                                                          │
│ 4. select_shard_files(1, 1)                             │
│    → 全部文件分配到 Shard 3                             │
│                                                          │
│ 5. run_excluded_tests_via_pytest()                      │
│    → 直接使用 pytest 执行 (不通过 run_test.py)          │
│    → NUM_PARALLEL_PROCS=8                               │
│    → pytest-xdist 并行执行                              │
└─────────────────────────────────────────────────────────┘
```

### 4.3 Regular Shard (4-10)

```
┌─────────────────────────────────────────────────────────┐
│ Shard 4-10: Regular Tests                               │
├─────────────────────────────────────────────────────────┤
│ 1. get_tests_list_from_discover_tests()                 │
│    → TESTS 列表 (discover_tests.py)                     │
│                                                          │
│ 2. filter_tests_by_type("regular")                      │
│    → 排除 test/distributed/*                            │
│    → 排除 EXCLUDED_TESTS_PATTERNS                       │
│                                                          │
│ 3. apply_case_path_rules(whitelist, blacklist)          │
│    → whitelist: 过滤出 whitelist 中的测试               │
│    → blacklist: 排除 blacklist 中的测试                 │
│                                                          │
│ 4. select_shard_files(shard_index, shard_total)         │
│    → 按文件数量均分到 Shard 4-10                        │
│                                                          │
│ 5. run_tests_via_run_test()                             │
│    → 使用 run_test.py 执行                              │
│    → NUM_PARALLEL_PROCS=8                               │
└─────────────────────────────────────────────────────────┘
```

---

## 五、Whitelist/Blacklist 配置机制

### 5.1 配置文件

`test_upsteam/case_paths_ci.yml`

```yaml
whitelist:
  - test/distributed
  - test/fx
  - test/mobile
  - test/quantization
  - test/custom_backend
  - test/custom_operator
  - test/nn
  - test/test_nn.py
  - test/test_ops.py
  # ... 160 entries total

blacklist:
  - test/distributed/test_nccl.py
  - test/distributed/test_c10d_nccl.py
  - test/fx/test_shape_inference.py
  - test/test_bundled_images.py
  # ... 30 entries total
```

### 5.2 两层过滤机制（重要说明）

#### 设计意图

Whitelist 和 Blacklist **不是矛盾关系**，而是**两层过滤机制**：

| 层级 | 作用 | 目的 |
|------|------|------|
| **Whitelist** | 定义**可能执行的测试范围** | 粗粒度控制，指定测试目录/文件 |
| **Blacklist** | 从 Whitelist 结果中**排除特定测试** | 细粒度控制，排除不支持/有问题的测试 |

#### 执行顺序

```
测试发现 (TESTS列表 或 原始扫描)
    ↓
Whitelist 过滤 (只保留 whitelist 中的测试)
    ↓
Blacklist 过滤 (从 whitelist 结果中排除 blacklist 测试)
    ↓
最终执行列表 = Whitelist 结果 - Blacklist 结果
```

#### 为什么所有 Blacklist 条目都在 Whitelist 中

这是**正确的设计**，不是矛盾：

- Whitelist 定义了**所有可能在 NPU 上运行的测试集合**
- Blacklist 精确排除**已知不支持或有问题的测试**
- 最终结果 = **白名单减去黑名单**

如果 Blacklist 条目不在 Whitelist 中：
- 该测试本来就不会被执行（不在 whitelist 范围内）
- Blacklist 条目就变成了冗余配置

### 5.3 Blacklist 条目分类

| 类别 | 条目数 | 排除原因 |
|------|--------|----------|
| **distributed** | 17 | NPU 不支持的分布式功能（NCCL、spawn、FSDP mixed precision 等） |
| **fx** | 2 | `test_shape_inference.py` 和 `test_future.py` 在 NPU 上有问题 |
| **excluded_tests** | 4 | 被 EXCLUDED_TESTS_PATTERNS 匹配但 blacklist 排除 |
| **other** | 7 | CUDA/XPU/MPS/Dynamo/Export 等平台特定测试 |

### 5.4 Distributed Blacklist 详细列表

以下 17 个 distributed 测试被 blacklist 排除，不会在 Shard 1-2 执行：

```
test/distributed/launcher                           # 目录级排除
test/distributed/test_nccl.py                       # NCCL 不支持
test/distributed/test_c10d_nccl.py                  # NCCL 不支持
test/distributed/test_c10d_ucc.py                   # UCC 不支持
test/distributed/test_c10d_spawn.py                 # spawn 不支持
test/distributed/test_distributed_spawn.py          # spawn 不支持
test/distributed/test_symmetric_memory.py           # symmetric memory 不支持
test/distributed/rpc/cuda/test_tensorpipe_agent.py  # CUDA RPC 不支持
test/distributed/fsdp/test_fsdp_mixed_precision.py  # FSDP mixed precision 不支持
test/distributed/fsdp/test_fsdp_comm_hooks.py       # FSDP comm hooks 不支持
test/distributed/_composable/fsdp/test_fully_shard_mixed_precision.py
test/distributed/_composable/fsdp/test_fully_shard_logging.py
test/distributed/tensor/test_matrix_ops.py          # distributed tensor 不支持
test/distributed/algorithms/quantization/test_quantization.py
test/distributed/bin/test_script.py                 # script 测试不支持
test/distributed/elastic/multiprocessing/bin/test_script.py
test/distributed/test_c10d_functional_native.py
```

### 5.5 Excluded Shard Blacklist 详细列表

以下 6 个测试在 EXCLUDED_TESTS_PATTERNS 中，但被 blacklist 排除，不会执行：

```
test/test_bundled_images.py           # blacklist 排除
test/test_cpp_extensions_aot.py       # blacklist 排除
test/custom_operator/test_custom_ops.py  # blacklist 排除
test/fx/test_future.py                # blacklist 排除
test/fx/test_shape_inference.py       # blacklist 排除
test/mobile/test_lite_script_module.py  # blacklist 排除
```

---

## 六、执行命令差异

### 6.1 Distributed/Regular Shard

使用 `run_test.py` 执行：

```bash
python run_test.py -i test1 test2 test3 -v --continue-through-error
```

环境变量：
- `NUM_PARALLEL_PROCS`: 文件级并行数

### 6.2 Excluded Shard

直接使用 `pytest` 执行：

```bash
python -m pytest --color=no -ra --tb=short \
    --continue-on-collection-errors \
    --junitxml=shard_3_excluded_pytest.xml \
    -p pytest_disabled_testcases_plugin \
    -n=8 --dist=loadfile \
    test/fx/test_common_passes.py test/mobile/test_bytecode.py ...
```

环境变量：
- `NPU_DISABLED_TESTCASES_JSON`: 禁用测试用例文件路径
- `NPU_DISABLED_TESTCASES_REPORT`: 禁用测试报告文件路径
- `PYTORCH_TEST_NPU`: "1"
- `TORCH_DEVICE_BACKEND_AUTOLOAD`: "1"

---

## 七、分片文件选择算法

### 7.1 连续范围分配

使用连续范围分配（而非轮询），确保同一目录的文件分布在相邻 Shard：

```python
def select_shard_files(test_files: List[str], shard: int, num_shards: int) -> List[str]:
    shard_index = shard - 1  # 转为 0-indexed
    total_files = len(test_files)
    
    base_size = total_files // num_shards
    remainder = total_files % num_shards
    
    # 前面的 Shard 多分配一个文件
    if shard_index < remainder:
        start = shard_index * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (shard_index - remainder) * base_size
        end = start + base_size
    
    return test_files[start:end]
```

### 7.2 分配示例

假设 100 个文件，10 个 Shard：

| Shard | 文件范围 | 文件数 |
|-------|---------|--------|
| 1 | 0-10 | 10 |
| 2 | 10-20 | 10 |
| 3 | 20-30 | 10 |
| ... | ... | ... |
| 10 | 90-100 | 10 |

---

## 八、关键文件路径

| 文件 | 作用 |
|------|------|
| `.github/scripts/run_npu_test_shard.py` | 分片执行主脚本 |
| `test_upsteam/case_paths_ci.yml` | Whitelist/Blacklist 配置 |
| `test_upsteam/disabled_testcases.json` | 禁用测试用例配置 |
| `test_upsteam/CRASHED.yml` | 崩溃测试文件配置 |
| `tools/testing/discover_tests.py` | TESTS 列表生成（上游） |
| `test/run_test.py` | 测试执行入口（上游） |

---

## 九、配置修改指南

### 9.1 添加新测试到执行范围

在 `case_paths_ci.yml` 的 `whitelist` 中添加：

```yaml
whitelist:
  - test/new_test_directory  # 添加目录
  - test/test_new_feature.py  # 添加单个文件
```

### 9.2 排除已知有问题的测试

在 `case_paths_ci.yml` 的 `blacklist` 中添加：

```yaml
blacklist:
  - test/test_problematic.py  # 排除单个文件
  - test/problematic_directory  # 排除整个目录
```

### 9.3 添加新的 Excluded Shard 测试

在 `run_npu_test_shard.py` 的 `EXCLUDED_TESTS_PATTERNS` 中添加：

```python
EXCLUDED_TESTS_PATTERNS = [
    # ... existing patterns
    "test/new_blocklisted_directory",  # 添加目录
    "test/test_new_blocklisted.py",    # 添加单个文件
]
```

---

## 十、常见问题

### Q1: 为什么某些测试不执行？

检查以下位置：
1. 是否在 `blacklist` 中
2. 是否不在 `whitelist` 中
3. 是否被 `disabled_testcases.json` 禁用

### Q2: 如何查看某个 Shard 执行了哪些测试？

查看测试报告目录：
- `shard_{N}_planned_test_files.txt`: 计划执行的测试文件
- `shard_{N}_excluded_test_files.txt`: 被排除的测试文件
- `shard_{N}_stats.json`: 执行统计

### Q3: Excluded Shard 为什么用 pytest 而不是 run_test.py？

因为 `run_test.py` 只能执行 TESTS 列表中的测试，而 Excluded Shard 的测试不在 TESTS 列表中。

---

## 十一、版本信息

- 文档版本: 1.0
- 适用版本: PyTorch 2.7.1 + torch-npu
- 更新日期: 2026-04-14