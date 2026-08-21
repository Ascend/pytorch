# NPUGraph_EX

## Overview

NPUGraph_EX is a lightweight, high-performance graph backend that integrates the graph sinking and scheduling capabilities of ACLGraph, and applies NPU-friendly graph optimizations and compilation cache reuse on top of PyTorch FX graphs, further accelerating the compilation and execution of LLMs on NPUs.

The core advantages of the NPUGraph_EX backend:

- **Graph offloading and scheduling**: Based on ACLGraph, it enables static graph capture and replay of NPU operations, eliminating kernel launch overhead.
- **FX graph optimization**: Performs NPU-friendly graph optimizations at the PyTorch FX graph level, reducing redundant computation and memory access.
- **Compilation cache reuse**: Supports caching and reuse of compilation results, avoiding the overhead of repeated compilation.
- **Service framework integration**: Enables fast and seamless integration with mainstream service frameworks, facilitating LLM inference deployment.

## Application Scenario

The NPUGraph_EX backend is suitable for LLM inference scenarios. It further accelerates compilation and execution through graph optimization and cache reuse, and integrates quickly with mainstream service frameworks.

## How to Enable

```python
compiled_model = torch.compile(model, backend="npugraph_ex")
```

For the compilation options (`options` parameter) supported by NPUGraph_EX and detailed usage guidance, see the [npugraph_ex backend](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/zh/npugraph_ex/npugraph_ex.md) in *TorchAir*.

## Call Example

```Python
import torch
import torch.nn as nn

# 1. Define the model
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 2. Compile the model (specify the npugraph_ex backend)
model = SimpleMLP().npu()
compiled_model = torch.compile(
    model,
    backend="npugraph_ex"  # Core: Enable NPUGraph_EX optimization
)

# 3. Training/Inference
input_data = torch.randn(32, 128).npu()

for _ in range(1000):  # High iteration count scenario (replay benefits are more obvious)
    output = compiled_model(input_data)
```

## Constraints

1. Optimizers are typically not captured in the graph. The optimizer's `step()` involves Python-side dynamic logic (such as learning rate scheduling, gradient accumulation, and adaptive update rules), which is difficult for static graphs to capture.
2. When using NPUGraph_EX, determine whether operators need to be updated during replay. If updates are required, enable the update mechanism.
