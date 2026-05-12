# Inductor

## 简介

Inductor是`torch.compile()`的默认后端编译器，通过"Dynamo前端图捕获+Inductor后端优化+高性能算子生成"的协同方式，在不改变模型逻辑的前提下自动进行算子融合和代码生成，显著提升训练或推理的吞吐量。

Inductor后端包含如下核心组件：

**表1** Inductor核心组件

| 组件 | 定位 | 作用 |
|------|------|------|
| Dynamo | 前端编译器（代码转换器） | TorchDynamo能够JIT（即时）将用户的eager（动态图）代码编译为FX Graph（PyTorch的中间表示），进而交给Inductor进行编译，最终生成优化过后的底层机器代码。 |
| Inductor | 后端编译器（高效代码生成器） | 具备基于多种模式（包括Triton/MLIR/DVM）的自动生成高性能算子能力，能够显著减少开发者手动设计Tiling、管理内存等工作量。支持算子融合等图优化策略，通过减少内存访问次数来提升性能。 |

## 算子编译器

Inductor后端支持三种算子编译器，可根据场景选择：

| 编译器 | 模式 | 说明 |
|--------|------|------|
| Triton-Ascend | 默认模式 | 基于Triton-Ascend生成融合算子，是Inductor后端的默认选择。详细介绍参考[Triton-Ascend官方仓库](https://gitcode.com/Ascend/triton-ascend)。 |
| Torch-MLIR | MLIR模式 | 基于Torch-MLIR生成融合算子。详细介绍参考[Torch-MLIR官方仓库](https://github.com/llvm/torch-mlir)。 |
| DVM | DVM模式 | 基于DVM生成融合算子。详细介绍参考[DVM官方仓库](https://gitcode.com/mindspore/dvm/tree/master)。 |

## 使用场景

Inductor后端以降低Python开销和kernel启动开销为核心，适合迭代次数多、单步计算量中等的场景。

**Triton模式（默认）**：通过`torch.compile(backend="inductor")`使能，基于Triton-Ascend生成融合算子，适用于大多数场景。

**MLIR模式**：通过`torch.compile(backend="inductor", options={"npu_backend": "mlir"})`使能，基于Torch-MLIR生成融合算子，需额外安装Torch-MLIR依赖包。

**DVM模式**：通过`torch.compile(backend="inductor", options={"npu_backend": "dvm"})`使能，基于DVM生成融合算子。

## 使用指导

> [!NOTICE]
>
> Inductor后端需安装最新版本的Triton-Ascend依赖包，具体可参考[Triton-Ascend说明文档](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md)。<br>
> Inductor后端使用MLIR模式时需额外安装Torch-MLIR依赖包，可以在[Torch-MLIR归档地址](https://repo.oepkgs.net/ascend/pytorch/vllm/torch/)下载。

接口原型：

```python
def compile(model, *, fullgraph=False, dynamic=None, backend="inductor", mode=None, options=None, disable=False)
```

Inductor后端支持的编译选项（`options`参数）包括：

- `triton.cudagraphs`：Triton相关配置
- `trace.enabled`：跟踪开关
- `enable_shape_handling`：形状处理配置
- `npu_backend`：指定算子编译器（`"mlir"`或`"dvm"`，默认为Triton）

该接口详情可参考原生[torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)。

## 使用样例

```Python
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
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="reduce-overhead"
)
# 如果需要指定算子编译器，加上选项options={"npu_backend": "mlir"}或{"npu_backend": "dvm"}

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

## 约束说明

1. 优化器（optimizer）通常不入图，优化器的`step()`包含Python侧动态逻辑（如学习率调度、梯度累积、自适应更新规则），难以被静态图捕获。
2. `torch.compile(backend="inductor")`支持动态形状，但会触发重新编译（增加开销），建议尽量固定形状。

## Triton（默认模式）

Triton是Inductor的默认算子编译器，通过`torch.compile(backend="inductor")`使能。

- 基于**Triton-Ascend**自动生成融合算子，适用于大多数场景
- 显著减少开发者手动设计Tiling、管理内存等工作量
- 支持算子融合等图优化策略，通过减少内存访问次数来提升性能
- 需安装最新版本的[Triton-Ascend依赖包](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md)

## MLIR

MLIR是Inductor的可选算子编译器，通过`torch.compile(backend="inductor", options={"npu_backend": "mlir"})`使能。

- 基于**Torch-MLIR**生成融合算子
- 需额外安装Torch-MLIR依赖包，可以在[Torch-MLIR归档地址](https://repo.oepkgs.net/ascend/pytorch/vllm/torch/)下载
- 适用于特定场景或需要MLIR中间表示进行进一步优化的场景

## DVM

DVM是Inductor的可选算子编译器，通过`torch.compile(backend="inductor", options={"npu_backend": "dvm"})`使能。

- 基于**DVM**生成融合算子
- 详细介绍参考[DVM官方仓库](https://gitcode.com/mindspore/dvm/tree/master)

## option配置

Inductor支持的编译选项（`options`参数）：

| 选项 | 说明 |
|------|------|
| `triton.cudagraphs` | Triton相关配置 |
| `trace.enabled` | 跟踪开关 |
| `enable_shape_handling` | 形状处理配置 |
| `npu_backend` | 指定算子编译器（`"mlir"`或`"dvm"`，默认Triton） |

### reduce-overhead（NPUGraph Tree）

`mode="reduce-overhead"`是降低开销的优化策略，核心对应**NPUGraph Tree**的逻辑：

- **动态形状路由**：管理多个有关联的NPUGraphs，让NPUGraph的优化收益能覆盖动态形状场景，而非仅局限于固定形状
- **子图内存优化**：优化段图场景多个子图的内存使用
- **工作原理**：在多次不同形状的输入下，Tree会缓存多个捕获的图版本，根据输入形状自动路由到最匹配的子图进行重放

```Python
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="reduce-overhead"  # 启用NPUGraph Tree逻辑
)
```

关键源文件（torch_npu仓库内）：

- `torch_npu/npu/_graph_tree.py` — NPUGraphTreeManager、GraphNode、WarmupNode
- `torch_npu/utils/_graph_tree.py` — NpugraphsBackend、后端注册
