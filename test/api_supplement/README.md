# Ascend/pytorch API一致性测试用例补充规范

## 1. 背景与目标

当前 PyTorch 与 Ascend/pytorch 中存在大量 API 未补充测试用例。为系统化补齐覆盖，需在仓库 `test/` 目录下**单独新增一个目录**，专门用于新增 API 测试用例，要求覆盖**功能、内存、确定性**等维度，目录按 `torch.nn.XX`、`torch.jit.xx` 等命名空间逐级创建，最终**每个 API 对应一个测试文件**。

同时要求添加标签或标记，使这些用例**不在常规 CI 中实际执行**，仅在全量回归或指定的 Nightly 任务中触发。

## 2. 目录规范

- 顶层目录建议为 `test/api_supplement/`，并提供 README 说明用途和运行方式。
- 按 API 公开命名空间逐级建目录，将点号 `.` 替换为路径分隔符 `/`，目录和文件名不得使用点号。
- 类方法目录应包含模块名（如适用）、类名和方法名；继承方法按实际被看护的 API 对象确定路径。
- 同一 API 的功能、内存、确定性等用例放在同一文件中，不按看护维度拆分文件。

例如，`torch.nn.functional.relu` 与 `torch.jit.script` 的目录结构如下：

```text
test/
└── api_supplement/
    ├── README.md
    └── torch/
        ├── nn/
        │   └── functional/
        │       └── test_torch_nn_functional_relu.py
        └── jit/
            └── test_torch_jit_script.py
```

## 3. 命名规范

### 3.1 文件命名

文件名采用 `test_<api_full_name_underscored>.py`，即 API 全名中的点号替换为下划线。类名等标识保留公开大小写，不强制转为小写。

### 3.2 测试方法命名

测试方法使用小写字母和下划线，采用 `test_<api_name>[_<dimension>...]_<scenario>` 格式。`scenario` 描述 dtype、shape、参数或场景；常规功能用例不添加额外维度字段。

特殊看护维度必须使用以下标准字段：

| 看护维度 | 命名字段 | 示例 |
| --- | --- | --- |
| 反向 | `backward` | `test_relu_backward_fp32` |
| 内存 | `memory` | `test_relu_memory_repeated_calls` |
| 确定性 | `deterministic` | `test_relu_deterministic_fixed_seed` |
| 边界值 | `boundary` | `test_relu_boundary_empty_tensor` |
| 小值域 | `small` | `test_relu_small_fp16` |

多维度字段按 `backward`、`memory`、`deterministic`、`boundary`、`small` 的顺序组合，不得使用 `mem`、`bwd`、`det`、`edge` 等缩写。

## 4. 用例编写与看护要求

### 4.1 基本要求

- 使用 `unittest`、仓库提供的 `TestCase` 和 `run_tests()`，禁止使用 pytest API。
- 使用 `self.device_name` 等设备抽象，不得在业务逻辑中硬编码设备；用例应具备 NPU/CUDA 可移植性。
- 数值或状态结果应与 CPU Golden 或明确的原生契约对比，浮点结果使用仓库认可的精度比较方法。
- 不得仅验证接口可调用、不报错、shape/dtype 正确或数值有限，必须验证 API 的实际语义。

### 4.2 最小示例

以下示例参考 `torch/optim` 用例，演示统一触发标记、NPU 优先/GPU 回退和 CPU Golden 对比。示例仅展示最小结构，正式提交仍需补齐适用的看护维度。

```python
# -*- coding: utf-8 -*-
"""API: torch.nn.functional.relu
覆盖：正常输入的功能正确性，NPU/GPU 与 CPU Golden 对比。
未覆盖：其他维度仅作示例，正式用例需按规范补齐或说明原因。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Set RUN_SUPPLEMENT_API_TESTS=1 to run supplement API tests"

DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

if DEVICE_NAME == "npu":
    from torch_npu.testing.testcase import TestCase, run_tests
else:
    from unittest import TestCase

    def run_tests():
        unittest.main()


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRelu(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_relu_fp32_1d(self):
        cpu_input = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        expected = torch.relu(cpu_input)
        actual = torch.relu(cpu_input.to(self.device))
        torch.testing.assert_close(actual.cpu(), expected)


if __name__ == "__main__":
    run_tests()
```

未检测到可用的 NPU 或 GPU 时，示例直接抛出 `RuntimeError`，不得回退 CPU。

### 4.3 覆盖维度

每个 API 至少覆盖功能；其他维度按 API 特性补充。不适用或未覆盖时必须说明原因。

| 维度 | 核心要求 |
| --- | --- |
| 功能 | 覆盖正常输入、适用的 dtype/shape、关键参数及模块的 training/eval 模式。 |
| 内存 | 检查连续调用后的内存趋势或 in-place 存储复用，避免明显泄漏或异常占用。 |
| 确定性 | 固定输入、设备和随机种子，多次执行结果一致或满足精度要求。 |
| 反向 | 对支持反向传播的 API，验证梯度 shape、dtype 和数值。 |
| 边界值 | 覆盖适用的空输入、极值、边界 shape 和参数边界。 |
| 小值域 | 对存在精度或溢出风险的 API，覆盖接近零等小值域输入。 |

### 4.4 API 契约与语义

- **参数与场景**：明确 API 全名和签名，覆盖默认/显式传参、参数类型、枚举选项、正常及异常场景。异常行为必须符合原生语义，不得凭空要求抛异常。
- **覆盖矩阵**：分析空/非空、枚举选项、参数类型、传参与不传参、等价类/边界值、正常传参、异常传参、混合设备类型。各项标记“已覆盖”“不适用”或“未覆盖”，并关联实际测试方法。
- **状态与计算**：有状态 API 应验证调用前后状态、返回值、重复调用及引用关系；计算类 API 应验证实际数值、状态更新或理论结果。
- **序列化与 Hook**：按适用情况覆盖序列化结构、保存加载往返、加载后连续执行，以及 Hook 的触发、顺序、参数、注销和非法回调。
- **继承与版本**：明确看护基类、子类或重载行为，不得以基类用例代替所有子类看护；版本相关行为以目标 PyTorch 版本为准并注明约束。
- **设备差异**：专用设备不可用时，只能跳过确实无法执行的场景并说明原因，设备无关行为和可验证的异常契约仍应保留。

### 4.5 文件头说明

文件头使用简体中文列出测试目的、API 全名与签名、覆盖矩阵、对应测试方法、未覆盖项及原因。说明必须与实际用例一致，不得标记未实际执行的覆盖项。

## 5. 合入规范

提交前完成以下自检：

- [ ] 目录、文件名和测试方法命名符合规范，每个 API 对应一个独立文件。
- [ ] 覆盖维度完整，文件头矩阵与实际用例一致，不适用和未覆盖项均有原因。
- [ ] 已按 API 特性验证功能、状态、数值、序列化、Hook、继承及版本相关语义。
- [ ] NPU/GPU 环境具备时分别完成全量验证；缺少环境、失败和跳过项均如实说明。
- [ ] PR 中提供覆盖清单、测试命令、执行环境和实际结果，不得仅提供收集成功结果。

## 6. 触发规范

### 6.1 常规 CI 与全量回归（待接入）

新增用例通过 `RUN_SUPPLEMENT_API_TESTS` 等统一标记控制。常规 CI 不设置该变量，用例不得实际执行，该目录下用例自动遵循此规则；全量回归或指定 Nightly 任务设置该变量为 `1`，确保用例真正运行而非仅被收集。

### 6.2 Nightly 分层执行

Nightly 提供以下执行范围，未指定时默认执行 `full`：

| 范围 | 执行要求 |
| --- | --- |
| `full` | 默认执行全部非黑名单用例，包含白名单用例且不得重复执行。 |
| `whitelist` | 批量执行高优 API，支持单独或按需触发；名单应精确匹配并记录调整原因。 |
| `blacklist` | 默认全量排除慢用例和特殊用例，分别通过慢用例任务或满足环境条件的专用任务执行。 |

统一要求：

- 批量策略按目录、API、看护维度或历史耗时稳定分组，并利用标准命名字段筛选和统计。
- 黑名单优先级高于白名单；每个条目记录用例全名、原因、执行方式和负责人，慢用例尽量附历史耗时。
- 黑名单用例不得永久失去看护，特殊用例不得在环境不满足时进入默认全量。
- 各批次输出计划数、实际执行数、通过数、失败数和跳过数；所有跳过项必须有明确原因，单批失败不得导致其他批次结果丢失。
