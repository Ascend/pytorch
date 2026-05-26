# torch.compile 概述

`torch.compile()` 是 PyTorch 2.0 推出的核心编译接口，Ascend Extension for PyTorch 从 2.6.0 版本开始支持。

## 工作流程

```text
用户代码 → Dynamo 前端（eager → FX Graph）→ 后端编译优化 → 执行
```

**Dynamo 前端**负责将 eager 代码 JIT 捕获为 FX Graph 中间表示；**后端**负责对 FX Graph 进行优化并生成最终可执行的代码。不同后端采用不同的优化策略，适用不同场景。

概念辨析、后端关系全景图见 [NPUGraph 与 torch.compile 概念总览](./pytorch_graph_desc.md)。

## 后端选择

通过 `torch.compile(backend=...)` 指定，同一层级、互斥选择：

| 后端 | 使能方式 | 核心机制 | 适用场景 |
|------|----------|---------|---------|
| **inductor**（默认） | `backend="inductor"` | 算子融合 + 代码生成（Triton / MLIR / DVM） | 大多数场景，不确定时首选 |
| **npugraphs** | `backend="npugraphs"` | ACLGraph 图下沉，一次捕获多次重放，消除 kernel 启动开销 | kernel 调用频繁、CPU 调度密集 |
| **npugraph_ex** | `backend="npugraph_ex"` | ACLGraph 图下沉 + FX 图优化 + 编译缓存复用 | 大模型推理部署 |
| **aot_eager** | `backend="aot_eager"` | 不做优化，仅验证图捕获正确性 | 调试、基线性能对比 |

此外 `mode="reduce-overhead"` 启用 **NPUGraph Tree** 逻辑——在 inductor 融合算子的基础上，管理多个 NPUGraph 子图，根据输入自动路由，让优化收益覆盖动态形状场景。

## 接口说明

> [!NOTICE]
> Inductor 后端需安装 [Triton-Ascend](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md)；MLIR 模式需额外安装 [Torch-MLIR](https://repo.oepkgs.net/ascend/pytorch/vllm/torch/)。

**接口原型：**

```python
torch.compile(model, *, fullgraph=False, dynamic=None, backend="inductor",
              mode=None, options=None, disable=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | nn.Module | 必填 | 待编译的模型 |
| fullgraph | bool | False | 是否强制整图编译 |
| dynamic | bool | None | 是否启用动态 shape 编译 |
| backend | str | `"inductor"` | 编译后端：`inductor`、`npugraphs`、`npugraph_ex`、`aot_eager` |
| mode | str | None | 编译模式：`None` 或 `"reduce-overhead"` |
| options | dict | None | 编译选项，详见各后端文档 |
| disable | bool | False | 关闭 torch.compile |

参考原生 [torch.compile 文档](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)。

## 使用样例

### Inductor 后端

```python
import torch
import torch_npu
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleMLP().npu()
compiled_model = torch.compile(model, backend="inductor", mode="reduce-overhead")

optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

x = torch.randn(32, 128).npu()
y = torch.randint(0, 10, (32,)).npu()

for _ in range(100):
    output = compiled_model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### NPUGraphs 后端

```python
import torch
import torch_npu
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleMLP().npu()
compiled_model = torch.compile(model, backend="npugraphs")

optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

x = torch.randn(32, 128).npu()
y = torch.randint(0, 10, (32,)).npu()

for _ in range(1000):
    output = compiled_model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### NPUGraph_EX 后端

```python
import torch
import torch_npu
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleMLP().npu()
compiled_model = torch.compile(model, backend="npugraph_ex")

x = torch.randn(32, 128).npu()

for _ in range(1000):
    output = compiled_model(x)
```

## 约束说明

1. 优化器的 `step()` 包含 Python 侧动态逻辑（学习率调度、梯度累积等），通常不入图。
2. `backend="npugraphs"` 底层使用静态内存模型，输入形状变化会触发 Graph Tree 重录制（有开销但不会报错）；固定形状可获得最佳性能。
3. ACLGraph 仅支持 aclnn 算子，非 aclnn 算子需 fallback 到 eager 执行。
4. replay 时如需动态更新算子参数（如 FlashAttention 的序列长度），可通过 `update` 机制实现。
