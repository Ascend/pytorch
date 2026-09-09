# NPUGraphs

## Overview

`npugraphs` is the graph capture backend for `torch.compile()`, built on NPUGraph (ACLGraph) technology. NPUGraphs captures a sequence of NPU operations (such as kernel calls and memory copies) into a static graph cached on the NPU device. This enables capture-once, replay-many execution, eliminating repeated kernel launch overhead.

When using this backend, Dynamo hands the FX Graph over to NPUGraph, which automatically offloads it into an ACLGraph for static graph replay optimization. There is no need to manually manage streams or the capture process. NPUGraphs is a high-level wrapper around the NPUGraph API that automates graph capture and replay. For finer-grained control (such as manual stream management, region-based capture, or safe subgraph capture under dynamic control flow), refer to the `NPUGraph` class, `graph()` context manager, and `make_graphed_callables` API in the [torch_npu.npu.NPUGraph](../framework_feature_guide_pytorch/pytorch_npugraph_desc.md) documentation.

## Applicable Scenarios

- Models with fixed input shapes and frequent kernel calls.
- High-iteration training or inference tasks that require complete elimination of launch overhead.

## How to Enable

```python
compiled_model = torch.compile(model, backend="npugraphs", options=None)
```

`options` parameter description:

| Option | Description |
| - | - |
| `triton.cudagraphs` | Triton-related configuration |
| `trace.enabled` | Trace switch |
| `enable_shape_handling` | Shape handling configuration |
| `npu_backend` | Specifies the operator compiler (`"mlir"` or `"dvm"`, default Triton) |

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
compiled_model = torch.compile(
    model,
    backend="npugraphs"
)

fixed_input = torch.randn(32, 128).npu()
fixed_target = torch.randint(0, 10, (32,)).npu()

for _ in range(1000):
    output = compiled_model(fixed_input)
```

## Constraints

1. **Input shapes must be fixed**: After capture, batch_size, sequence length, and so on cannot be modified.
2. **Only NN operators are supported**: All operators must be aclnn operators to be included in the graph.
3. For dynamic shape support, consider using the `reduce-overhead` mode (NPUGraph Tree) of the Inductor backend.
4. The `npu_fusion_attention_v3` interface does not currently support graph capture in TND format.
