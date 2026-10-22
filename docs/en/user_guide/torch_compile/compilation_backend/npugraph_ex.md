# NPUGraph_EX

## Introduction

NPUGraph_EX is a lightweight, high-performance graph backend. It integrates the ACLGraph graph sinking and scheduling capability, and superimposes NPU-affinity graph optimization and compilation cache reuse capabilities on the PyTorch FX graph, further accelerating the compilation and running of large models on NPUs.

The core advantages of the NPUGraph_EX backend:

- **Graph sinking scheduling**: Based on ACLGraph, implements static graph capture and replay of NPU operations, eliminating kernel launch overhead.
- **FX graph optimization**: Performs NPU-affinity graph optimization at the PyTorch FX graph level, reducing redundant computation and memory access.
- **Compilation cache reuse**: Supports caching and reuse of compilation results, avoiding the overhead of repeated compilation.
- **Service framework integration**: Integrates quickly and seamlessly with mainstream service frameworks, facilitating large model inference deployment.

## Application Scenarios

The NPUGraph_EX backend is suitable for large model inference scenarios. It further accelerates compilation and running through graph optimization and cache reuse, and integrates quickly with mainstream service frameworks.

## How to Enable

```python
compiled_model = torch.compile(model, backend="npugraph_ex")
```

For the compilation options supported by NPUGraph_EX (the `options` parameter) and detailed usage guidance, refer to [npugraph_ex backend](https://gitcode.com/Ascend/torchair/blob/26.2.0/docs/en/npugraph_ex/npugraph_ex.md) in *TorchAir*.

## Calling Example

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
    backend="npugraph_ex"  # Key: enable NPUGraph_EX optimization
)

# 3. Training/inference
input_data = torch.randn(32, 128).npu()

for _ in range(1000):  # High-iteration scenarios (repeated runs yield more obvious benefits)
    output = compiled_model(input_data)
```

## Constraints

1. Optimizers are usually not included in the graph. The optimizer's `step()` contains Python-side dynamic logic (such as learning rate scheduling, gradient accumulation, and adaptive update rules), which is difficult to capture in a static graph.
2. When using NPUGraph_EX, determine whether operators need to be updated during replay. If so, enable the update mechanism.
