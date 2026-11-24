# Ascend/pytorch CI 流水线结构参考

## 仓库信息

- **GitHub 仓库**: https://github.com/Ascend/pytorch
- **GitCode 镜像**: https://gitcode.com/Ascend/pytorch
- **CI 工作流目录**: `.github/workflows/`

## 工作流类型

| 工作流 | 文件 | 触发方式 | 特点 |
|---|---|---|---|
| Nightly Build and Test | `nightly.yml` | push 到 ci-test 分支 | 全量测试，无 skip_list 过滤 |
| PyTorch CI Trigger | `pytorch_ci_trigger.yml` | repository_dispatch (PR) | PR 流水线，有 skip_list 过滤 |

## 测试分片结构

每个流水线将测试分为两个类型：
- **regular** (reg-1 ~ reg-5): 常规测试，5 个分片
- **distributed** (dist-1 ~ dist-5): 分布式测试，5 个分片

每个分片有 32 个并行 worker。

## 制品（Artifacts）清单

### 测试报告制品

| 制品名 | 内容 | 用途 |
|---|---|---|
| `npu-full-test-summary` | 全局测试摘要 JSON | 快速定位失败分片和用例数 |
| `test-reports-reg-N` | reg-N 分片的详细报告 | 用例级失败分析 |
| `test-reports-dist-N` | dist-N 分片的详细报告 | 分布式用例失败分析 |
| `cases-shards` | 测试用例分片信息 | 确认用例分片分配 |

### 日志制品

| 制品名 | 内容 | 用途 |
|---|---|---|
| `ascend-cann-logs-reg-N` | reg-N 分片的 CANN 日志 | 算子层执行日志分析 |
| `ascend-cann-logs-dist-N` | dist-N 分片的 CANN 日志 | 分布式算子日志分析 |
| `collection-errors` | 采集阶段错误 | 排查用例未采集问题 |

### 构建制品

| 制品名 | 内容 | 用途 |
|---|---|---|
| `build-artifacts-<N>` | 构建产物 | 确认构建参数 |
| `packages-pytorch-<hash>` | 编译好的 torch_npu 包 | 确认编译版本 |
| `pytorch-source-<hash>.tar.gz` | pytorch 源码包 | 确认上游 pytorch 版本 |

## 测试报告文件结构

### `npu-full-test-summary.json`

```json
{
  "overall_status": "FAILED",
  "docker_image": "quay.io/kerer/pytorch:torch-npu-test-aarch64-cann-a3-py3.10-torch-nightly",
  "pytorch_version": "nightly",
  "torch_npu_whl": "source-build",
  "totals": {
    "total": 48247,
    "passed": 48244,
    "failed": 3,
    "errors": 0,
    "skipped": 0,
    "timeout": 0,
    "duration": 7380.66
  },
  "shards": [
    {
      "shard": "reg-2",
      "shard_type": "regular",
      "status": "FAILED",
      "total": 9524,
      "passed": 9521,
      "failed": 3,
      "errors": 0
    }
  ]
}
```

### `shard_reg-N_cases.json`

```json
{
  "shard": "reg-2",
  "shard_type": "regular",
  "total_cases": 9524,
  "passed": 9521,
  "failed": 3,
  "cases": [
    {
      "nodeid": "test/nn/test_embedding.py::TestEmbeddingNNDeviceTypePRIVATEUSE1::test_xxx_npu_int64_int32_float32",
      "status": "failed",
      "duration": 13.25,
      "returncode": 1,
      "message": "AssertionError: Tensor-likes are not close!...",
      "file": "test/nn/test_embedding.py",
      "command": "/opt/conda/envs/py_3.10/bin/python -m pytest ...",
      "case_idx": 447
    }
  ]
}
```

### `cases_logs/reg-N_<IDX>_<test_name>.log`

每个用例的完整 pytest 日志，包含：
- 命令行（Command）
- NPU 设备号（NPU Device）
- 重试次数（Retry Count，如果有）
- STDOUT（pytest 输出）
- STDERR（CANN/allocator 警告等）
- RETRY HISTORY（重试历史，如果有）

## CANN 日志结构

```
ascend/
├── log/
│   ├── debug/
│   │   └── plog/
│   │       └── plog-<PID>_<TIMESTAMP>.log    # 算子执行日志
│   └── run/
│       └── device-<N>/
│           └── device-<PID>_<TIMESTAMP>.log  # 设备级运行日志
```

### plog 日志格式

```
[ERROR] OP(<PID>,python):<TIMESTAMP> [<source_file>:<line>][<module>][<function>][<PID>]
  errno[<error_code>] OpName:[<op_name>] <error_message>
```

典型错误类型：
- `CheckDtypeValid` — dtype 不支持（errno 161002）
- `CheckParams` — 参数检查失败（errno 561000）
- `GetWorkspaceSize` — workspace 获取失败
- `~UniqueExecutor` — executor 释放异常

## 关键子模块

| 子模块 | 路径 | 仓库 | 说明 |
|---|---|---|---|
| op-plugin | `third_party/op-plugin` | gitcode.com/ascend/op-plugin | 算子实现（opapi + aclops） |
| CANN runtime | `third_party/acl_src/runtime` | gitcode.com/cann/runtime | CANN 运行时 |
| CANN ge | `third_party/acl_src/ge` | gitcode.com/cann/ge | 图引擎 |
| graph-autofusion | `third_party/acl_src/graph-autofusion` | gitcode.com/cann/graph-autofusion | 自动融合 |
| torchair | `third_party/torchair/torchair` | gitcode.com/ascend/torchair | torchair 集成 |
| torch-mlir | `third_party/torch-mlir` | gitcode.com/gh_mirrors/to/torch-mlir | MLIR 集成（update=none） |
| dvm | `third_party/dvm/dvm` | gitcode.com/mindspore/dvm | DVM 后端 |

## op_plugin 算子双路径

每个算子在 op_plugin 中通常有两条实现路径：

```
op_plugin/ops/
├── opapi/           # 新路径，调用 aclnn 接口
│   └── EmbeddingBagNpuOpApi.cpp
└── aclops/          # 旧路径，调用 aclop 接口
    └── EmbeddingBagKernelNpu.cpp
```

`DO_COMPATIBILITY` 宏决定使用哪条路径：

```cpp
DO_COMPATIBILITY(aclnnEmbeddingBag, acl_op::_embedding_bag(...));
// 如果 CANN 库中存在 aclnnEmbeddingBag 函数地址 -> 走 opapi 路径
// 否则 -> fallback 到 aclop 路径
```

两条路径的实现细节可能不同（如空输入处理、内存初始化），这是回归分析的常见焦点。
