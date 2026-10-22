# AOT_Eager

## Introduction

`aot_eager` is one of the native `torch.compile()` backends of PyTorch. AOT_Eager compiles the FX Graph captured by `torch.compile` through the Ahead-of-Time (AOT) automatic differentiation mechanism, but ultimately falls back to Eager mode execution without performing additional operator fusion or code generation optimization.

The features of AOT_Eager are as follows:

- Does not perform operator fusion and does not generate Triton/DVM/Ascend C fused kernels.
- Preserves the complete graph structure and automatic differentiation information.
- Performance is similar to native Eager mode, with no obvious optimization gains.

For more details, see [AOTAutograd](https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_aot_compile.html).

## Usage Scenarios

- **Debugging and verification**: Confirms that the model executes correctly in `torch.compile` graph mode, ruling out issues introduced by Inductor or other optimization backends.
- **Compatibility testing**: Verifies whether graph capture and graph break behavior is normal.
- **Baseline comparison**: Compares performance with optimization backends such as Inductor and NPUGraphs.

## How to Enable

```python
compiled_model = torch.compile(model, backend="aot_eager")
```

## Calling Example

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
