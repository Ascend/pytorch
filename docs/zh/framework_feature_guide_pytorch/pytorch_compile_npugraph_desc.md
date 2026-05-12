# NPUGraphs

## 简介

`npugraphs`是`torch.compile()`的图捕获后端，基于NPUGraph(ACLGraph)技术实现。核心思想与[CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)一致——捕获一系列NPU操作（如kernel调用、内存拷贝）组成静态图，**一次捕获、多次复跑**，彻底避免重复的kernel启动开销。

使用该后端时，TorchDynamo将FX Graph交由NPUGraph自动下沉为ACLGraph静态图重放优化，无需手动管理Stream和捕获流程。

## 使能方式

```python
compiled_model = torch.compile(model, backend="npugraphs")
```

## 适用场景

- 输入形状固定、kernel调用频繁的模型
- 需要彻底消除启动开销的高迭代训练/推理任务

## 使用样例

```python
import torch
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
    backend="npugraphs"
)

fixed_input = torch.randn(32, 128).npu()
fixed_target = torch.randint(0, 10, (32,)).npu()

for _ in range(1000):
    output = compiled_model(fixed_input)
```

## 注意事项

### 使用约束

1. **必须固定输入形状**：捕获后无法修改batch_size、序列长度等
2. **仅支持NN算子**：所有算子必须为aclnn算子方可入图
3. 如需动态形状支持，请考虑使用Inductor后端的`reduce-overhead`模式（NPUGraph Tree）

### 与NPUGraph底层API的关系

`torch.compile(backend="npugraphs")`是NPUGraph底层API的高级封装，自动完成图捕获和重放流程。如需更精细的控制（如手动管理Stream、分区域捕获、动态控制流下的安全子图捕获等），请参考[NPUGraph](./pytorch_npugraph_desc.md)文档中的`NPUGraph`类、`graph()`上下文管理器和`make_graphed_callables` API。
