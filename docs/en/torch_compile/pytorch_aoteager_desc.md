# AOT_Eager

## Overview

`aot_eager` is one of the native `torch.compile()` backends in PyTorch. AOT_Eager compiles the FX Graph captured by `torch.compile` through the Ahead-of-Time (AOT) automatic differentiation mechanism, but ultimately falls back to Eager mode for execution, without performing additional operator fusion or code generation optimizations.

The characteristics of AOT_Eager are as follows:

- No operator fusion is performed, and no Triton/MLIR/DVM fused kernels are generated.
- The complete graph structure and automatic differentiation information are preserved.
- Performance is similar to native Eager mode, with no significant optimization benefits.

For more details, see [AOTAutograd](https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_aot_compile.html).

## Applicable Scenarios

- **Debugging and Verification**: Confirm whether the model executes correctly in `torch.compile` graph mode, and rule out issues introduced by Inductor or other optimization backends.
- **Compatibility Testing**: Verify whether graph capture and graph break behavior is normal.
- **Baseline Comparison**: Compare performance against optimization backends such as Inductor and NPUGraphs.

## How to Enable

```python
compiled_model = torch.compile(model, backend="aot_eager")
```

## Call Example

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
