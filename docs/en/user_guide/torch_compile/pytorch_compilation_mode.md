# Overview

`torch.compile()` is the core compilation interface introduced in PyTorch 2.0, which significantly accelerates model training and inference tasks through the approach of dynamic graph capture + static graph optimization + efficient code generation. Starting from TorchNPU 7.3.0, `torch.compile()` is supported, enabling automatic frontend graph capture and backend optimization with just a single line of code, making it suitable for fully automated compilation scenarios.

`torch.compile` includes the following core components:

**Table 1** Core Components

| Component | Function |
| ------------------- | ----------- |
| Dynamo Frontend | Dynamo can JIT (just-in-time) compile the user's eager code into FX Graph (PyTorch's intermediate representation). |
| Compilation Backend | Optimizes the FX Graph and generates the final executable code. |

## API Description

### API Prototype

```python
torch.compile(model, *, fullgraph=False, dynamic=None, backend="inductor",
              mode=None, options=None, disable=False)
```

### Parameter Description

| Parameter | Type | Default Value | Description |
| ------ | ------ | -------- | ------ |
| model | nn.Module | Mandatory | The model to be compiled |
| fullgraph | bool | False | Whether to force full-graph compilation |
| dynamic | bool | None | Whether to enable dynamic shape compilation |
| backend | str/Callable | `"inductor"` | Compilation backend: `inductor`, `npugraphs`, `npugraph_ex`, `aot_eager`, `TorchAir-GE backend (Callable)` |
| mode | str | None | Compilation mode: `None` or `"reduce-overhead"` (supported only by the `inductor` backend) |
| options | dict | None | Compilation options |
| disable | bool | False | Disable torch.compile |

For more parameter details, see [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

**Compilation Backend Description**

| Backend | Enabling Method | Core Mechanism | Applicable Scenario |
| ------ | ---------- | --------- | --------- |
| Inductor (default) | `backend="inductor"` | Operator fusion + code generation (Triton/MLIR/DVM) | Most scenarios. When uncertain, choose it first |
| NPUGraphs | `backend="npugraphs"` | ACLGraph graph sinking, capture once and replay multiple times, eliminating kernel launch overhead | Frequent kernel calls, CPU-intensive scheduling |
| NPUGraph_EX | `backend="npugraph_ex"` | ACLGraph graph sinking + FX graph optimization + compilation cache reuse | LLM inference deployment |
| AOT_Eager | `backend="aot_eager"` | No optimization, only verifies graph capture correctness | Debugging, baseline performance comparison |
| TorchAir-GE | `backend=torchair.get_npu_backend(...)` | Converts PyTorch FX graphs into computation graphs and performs graph compilation and execution via the GE graph engine | LLM inference deployment |
