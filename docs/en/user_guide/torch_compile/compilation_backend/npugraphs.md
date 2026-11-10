# NPUGraphs

## Introduction

`npugraphs` is the graph capture backend of `torch.compile()`, implemented based on NPUGraph (ACLGraph) technology. NPUGraphs captures a series of NPU operations (such as kernel calls and memory copies) into a static graph cached on the NPU device. It captures once and replays multiple times, avoiding repeated kernel launch overhead.

When using this backend, Dynamo hands the FX Graph to NPUGraph, which automatically sinks it into an ACLGraph static graph for replay optimization, without the need to manually manage Streams and the capture process. NPUGraphs is a high-level wrapper of the NPUGraph API, automatically completing the graph capture and replay process. For finer-grained control (such as manually managing Streams, region-based capture, and safe subgraph capture under dynamic control flow), refer to the `NPUGraph` class, the `graph()` context manager, and the `make_graphed_callables` API in the [torch_npu.npu.NPUGraph](../../../developer_notes/npugraph.md) documentation.

## Application Scenarios

- Models with fixed input shapes and frequent kernel calls.
- High-iteration training/inference tasks that require complete elimination of launch overhead.

## How to Enable

```python
compiled_model = torch.compile(model, backend="npugraphs", options=None)
```

Description of the `options` parameter:

| Option | Description |
| - | - |
| `triton.cudagraphs` | Triton-related configuration |
| `trace.enabled` | Tracing switch |
| `enable_shape_handling` | Shape handling configuration |
| `npu_backend` | Specifies the operator compiler (`"dvm"` or `"ascendc"`; Triton by default) |

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
compiled_model = torch.compile(
    model,
    backend="npugraphs"
)

fixed_input = torch.randn(32, 128).npu()
fixed_target = torch.randint(0, 10, (32,)).npu()

for _ in range(1000):
    output = compiled_model(fixed_input)
```

## Constraint Description

- **The input shape must be fixed**: After capture, batch_size, sequence length, and other attributes cannot be modified.
- **Only NN operators are supported**: All operators must be aclnn operators to be included in the graph.
- For dynamic shape support, consider using the `reduce-overhead` mode (NPUGraph Tree) of the Inductor backend.
