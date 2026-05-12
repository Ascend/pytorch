# NPUGraph_EX

## 简介

NPUGraph_EX是轻量化高性能图后端，融合了ACLGraph的图下沉调度能力，在PyTorch FX图上叠加亲和NPU的图优化和编译缓存复用等能力，进一步加速大模型在NPU上编译运行。

NPUGraph_EX后端的核心优势：

- **图下沉调度**：基于ACLGraph实现NPU操作的静态图捕获和复跑，消除kernel launch开销。
- **FX图优化**：在PyTorch FX图层面进行亲和NPU的图优化，减少冗余计算和内存访问。
- **编译缓存复用**：支持编译结果的缓存复用，避免重复编译带来的开销。
- **服务化框架对接**：与主流服务化框架快速、无缝地对接，便于大模型推理部署。

## 使用场景

NPUGraph_EX后端适用于大模型推理场景，通过图优化和缓存复用进一步加速编译运行，并与主流服务化框架快速对接。

使能方式：`torch.compile(backend="npugraph_ex")`

## 使用指导

接口原型：

```python
def compile(model, *, fullgraph=False, dynamic=None, backend="npugraph_ex", mode=None, options=None, disable=False)
```

NPUGraph_EX支持的编译选项（`options`参数）和详细使用指导请参考《PyTorch图模式使用(TorchAir)》中的 [npugraph_ex后端](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/npugraph_ex.md)。

该接口详情可参考原生 [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)。

## 使用样例

```Python
import torch
import torch.nn as nn

# 1. 定义模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 2. 编译模型（指定 npugraph_ex 后端）
model = SimpleMLP().npu()
compiled_model = torch.compile(
    model,
    backend="npugraph_ex"  # 核心：启用 NPUGraph_EX 优化
)

# 3. 训练/推理
input_data = torch.randn(32, 128).npu()

for _ in range(1000):  # 高迭代次数场景（复跑收益更明显）
    output = compiled_model(input_data)
```

## 约束说明

1. 优化器（optimizer）通常不入图，优化器的`step()`包含Python侧动态逻辑（如学习率调度、梯度累积、自适应更新规则），难以被静态图捕获。
2. 使用NPUGraph_EX时需要判断算子在replay时是否需要更新，如需更新，启用update机制。
