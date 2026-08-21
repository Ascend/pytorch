# TorchAir-GE Backend

## Overview

The TorchAir-GE backend converts the PyTorch FX computation graph into an intermediate representation (IR) and compiles and executes the computation graph through GE (Graph Engine).

Core advantages of the TorchAir-GE backend:

- **FX graph optimization**: Performs NPU-friendly graph optimization at the PyTorch FX graph level, reducing redundant computation and memory access.
- **Compilation cache reuse**: Supports caching and reuse of compilation results, avoiding the overhead of repeated compilation.

## Application Scenario

The TorchAir-GE backend is suitable for LLM inference scenarios, further accelerating compilation and execution through graph optimization and cache reuse, and enabling rapid integration with mainstream service-oriented frameworks.

## How to Enable

```python
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
compiled_model = torch.compile(model, backend=npu_backend)
```

For the compilation options supported by the TorchAir-GE backend (the `compiler_config` parameter) and detailed usage guidance, see [GE Graph Mode](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/en/ascend_ir/quick_start.md) in *TorchAir*.

## Call Example

```Python
import torch
import torch_npu
import torchair

# 1. Define the model
class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 2. Create the TorchAir-GE backend
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
    
# 3. Compile the model (using the TorchAir-GE backend)
model = SimpleMLP().npu()
compiled_model = torch.compile(model, backend=npu_backend)

# 4. Inference
input_data = torch.randn(32, 128).npu()

for _ in range(1000):
    output = compiled_model(input_data)
```

## Constraints

1. Before using the TorchAir-GE backend, ensure that the model runs correctly in Ascend NPU single-operator mode (Eager).
2. The script must first `import torch_npu`, then `import torchair` to function properly.
