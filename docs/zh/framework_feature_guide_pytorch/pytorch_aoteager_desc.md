# AOT_Eager

## 简介

`aot_eager`是PyTorch原生的`torch.compile()`后端之一。它的核心思路是将`torch.compile`捕获的FX Graph通过Ahead-of-Time（AOT）自动微分机制进行编译，但最终回退到Eager模式执行，不做额外的算子融合或代码生成优化。

## 使能方式

```python
compiled_model = torch.compile(model, backend="aot_eager")
```

## 适用场景

- **调试与验证**：确认模型在`torch.compile`的图模式下是否正确执行，排除Inductor或其他优化后端引入的问题
- **兼容性测试**：验证图捕获和图断裂（graph break）行为是否正常
- **基准对照**：与Inductor、NPUGraphs等优化后端做性能对比

## 特点

- 不做算子融合，不生成Triton/MLIR/DVM融合内核
- 保留完整的图结构和自动微分信息
- 性能与原生Eager模式相近，无明显优化收益

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
compiled_model = torch.compile(model, backend="aot_eager")

x = torch.randn(32, 128).npu()
output = compiled_model(x)
```

## 参考

- PyTorch原生[torch.compile backends](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [AOTAutograd说明](https://pytorch.org/docs/stable/torch.compiler_aot_autograd.html)
