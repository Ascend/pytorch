# NPUGraph_EX

## Overview

NPUGraph_EX is a lightweight, high-performance graph backend that integrates the graph sinking and scheduling capabilities of ACLGraph, and applies NPU-friendly graph optimizations and compilation cache reuse on top of PyTorch FX graphs, further accelerating the compilation and execution of LLMs on NPUs.

The core advantages of the NPUGraph_EX backend:

- **Graph offloading and scheduling**: Based on ACLGraph, it enables static graph capture and replay of NPU operations, eliminating kernel launch overhead.
- **FX graph optimization**: Performs NPU-friendly graph optimizations at the PyTorch FX graph level, reducing redundant computation and memory access.
- **Compilation cache reuse**: Supports caching and reuse of compilation results, avoiding the overhead of repeated compilation.
- **Service framework integration**: Enables fast and seamless integration with mainstream service frameworks, facilitating LLM inference deployment.

> [!NOTICE]
>
> - Before using the NPUGraph_EX backend, install the TorchNPU plugin and the packaging library for the corresponding Python version (version 21.3 or later). Some Python code of NPUGraph_EX depends on the packaging library for version comparison. If the library is missing, the first compilation fails with the error "ModuleNotFoundError: No module named 'packaging'", which is wrapped by PyTorch as "BackendCompilerFailed".
>   Run `python -m pip install "packaging>=21.3"` in the Python environment used to run the model.

## Application Scenario

The NPUGraph_EX backend is suitable for LLM inference scenarios. It further accelerates compilation and execution through graph optimization and cache reuse, and integrates quickly with mainstream service frameworks.

## How to Enable

```python
compiled_model = torch.compile(model, backend="npugraph_ex")
```

For the compilation options (`options` parameter) supported by NPUGraph_EX and detailed usage guidance, see the [npugraph_ex backend](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/en/npugraph_ex/npugraph_ex.md) in *TorchAir*.

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

- Optimizers are typically not captured in the graph. The optimizer's `step()` involves Python-side dynamic logic (such as learning rate scheduling, gradient accumulation, and adaptive update rules), which is difficult for static graphs to capture.
- When using NPUGraph_EX, determine whether operators need to be updated during replay. If updates are required, enable the update mechanism.
- The backend is mainly intended for online inference scenarios. Backward-process capture and random-number operator capture are not supported.
- The constraints are consistent with those of the native torch.cuda.CUDAGraph interface, for example, stream sync and dynamic control flow are not supported.
- Set the `fullgraph` parameter to `True` whenever possible. If the compiler encounters code that cannot be traced into a single computation graph (that is, a graph break), an error is raised.
- The `mode` parameter is not supported on Ascend NPUs. Pass feature configurations through the `options` parameter.
- Run `torch.compile(backend="aot_eager", fullgraph=True)` first to verify that the script can be traced into a graph. If this step fails, the script or operator itself does not meet the conditions for using `backend="npugraph_ex"`.

## Reference Documents

This page provides only the minimal example and general constraints of the NPUGraph_EX backend from the torch.compile perspective. The complete feature items supported by `options`, the specific constraints and configuration methods of each feature, troubleshooting methods, and API references are maintained in the "[npugraph_ex](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/en/npugraph_ex/quick_start.md)" chapter of *TorchAir*. See the following content.

- [npugraph_ex Quick Start](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/en/npugraph_ex/quick_start.md): usage constraints, torch.compile parameter descriptions in aclgraph mode, the basic/advanced/DFX feature list, troubleshooting, and feature extension.
- [Basic Features](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/en/npugraph_ex/basic/basic.md): force_eager, force_recapture, FX graph optimization passes, memory reuse, static kernel compilation, frozen parameters, collective communication graphs, and other features and their constraints.
- [Advanced Features](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/en/npugraph_ex/advanced/advanced.md): model compilation cache, multi-stream expression, AI Core and Vector Core core limiting, custom FX graph optimization passes, SuperKernel, and other features and their constraints.
- [DFX Features](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/en/npugraph_ex/dfx/dfx.md): graph compilation debug information saving, operator Data Dump, and multi-stream concurrent deadlock detection.
- [API Reference](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/en/npugraph_ex/api/api.md): torch.npu.npugraph_ex, compile_fx, CompilerConfig, register_replacement, and scope interfaces.
