# TorchAir-GE Backend

## Introduction

The TorchAir-GE backend converts PyTorch FX computation graphs into an intermediate representation (IR, Intermediate Representation), and implements the compilation and execution of computation graphs through GE (Graph Engine).

The core advantages of the TorchAir-GE backend:

- **FX graph optimization**: Performs NPU-affinity graph optimization at the PyTorch FX graph level, reducing redundant computation and memory access.
- **Compilation cache reuse**: Supports caching and reuse of compilation results, avoiding the overhead of repeated compilation.

## Application Scenarios

The TorchAir-GE backend is suitable for large model inference scenarios. It further accelerates compilation and running through graph optimization and cache reuse, and integrates quickly with mainstream service frameworks.

## How to Enable

```python
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
compiled_model = torch.compile(model, backend=npu_backend)
```

For the compilation options supported by the TorchAir-GE backend (the `compiler_config` parameter) and detailed usage guidance, refer to [GE Graph Mode](https://gitcode.com/Ascend/torchair/blob/26.2.0/docs/en/ascend_ir/quick_start.md) in *TorchAir*.

## Calling Example

```Python
import torch
import torch_npu
import torchair
import torch.nn as nn

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

1. Before using the TorchAir-GE backend, ensure that the model executes correctly in Ascend NPU single-operator mode (Eager).
2. In the script, `import torch_npu` must be executed before `import torchair` for normal use.
