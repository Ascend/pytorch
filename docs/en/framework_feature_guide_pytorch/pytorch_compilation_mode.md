# PyTorch Compilation Mode (torch.compile)

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-08T10:21:46.031Z pushedAt=2026-07-08T10:47:16.862Z -->

## Introduction

torch.compile() is the core optimization interface introduced in PyTorch 2.0+, which significantly accelerates model training and inference tasks through "eager mode capture + graph mode optimization + efficient code generation." Ascend Extension for PyTorch supports this feature in version 2.6.0 and above, providing users with three commonly used backend configuration options: torch.compile(backend="inductor"), torch.compile(backend="npugraphs"), and torch.compile(backend="npugraph_ex").

torch.compile() includes the following core components:

**Table 1** Core Components

| Component | Positioning | Role |
|--------------------|--|----------------------------------------------------------------------------------------------------------|
| Dynamo | Frontend compiler (code converter) | TorchDynamo can JIT-compile users' eager mode code into FX Graph (PyTorch's intermediate representation), which is then passed to other lowering compilers (such as Inductor — the backend compiler responsible for converting graphs into efficient machine code) for compilation, ultimately generating optimized low-level machine code to achieve acceleration. |
| Inductor | Backend compiler (efficient code generator) | Capable of automatically generating high-performance operators based on multiple modes (including Triton/MLIR/DVM), significantly reducing the workload of manual tiling design and memory management. Supports graph optimization strategies such as operator fusion, improving performance by reducing the number of memory accesses. |
| NPUGraph (ACLGraph) | Hardware-level offloading optimization (NPU operation recording) | Captures a series of NPU operations (such as kernel calls and memory copies) to form a static graph cached on the NPU device; captured once and replayed multiple times, avoiding repeated kernel launch overhead. |
| NPUGraph_EX | Lightweight high-performance graph backend | Integrates ACLGraph's graph offloading scheduling capability, superimposing NPU-friendly graph optimizations and compilation cache reuse on PyTorch FX graphs to further accelerate compilation and execution of large models on NPUs. |
| NPUGraph Tree | Dynamic shape routing and subgraph management | Manages multiple related NPUGraphs, extending the optimization benefits of NPUGraph to cover dynamic shape scenarios rather than being limited to fixed shapes, and optimizing memory usage across multiple subgraphs in segmented graph scenarios. |

## Use Cases

Inductor backend: Enabled via `torch.compile(backend="inductor")`, focusing on reducing Python overhead and kernel launch overhead. Through the collaboration of Dynamo and Inductor, it automatically performs operator fusion and generation without altering the model logic, improving training or inference throughput. It is particularly suitable for scenarios with a high number of iterations and moderate per-step computation.

- Triton mode: The default mode of the Inductor backend, which generates fused operators based on Triton-Ascend. For more details about Triton-Ascend, see the [Triton-Ascend official repository](https://gitcode.com/Ascend/triton-ascend).
- MLIR mode: Enabled via `torch.compile(backend="inductor", options={"npu_backend": "mlir"})`, which generates fused operators based on Torch-MLIR. For more details about Torch-MLIR, see the [Torch-MLIR official repository](https://github.com/llvm/torch-mlir).
- DVM mode: Enabled via `torch.compile(backend="inductor", options={"npu_backend": "dvm"})`, which generates fusion operators based on DVM. For a detailed introduction to DVM, refer to the [DVM official repository](https://gitcode.com/mindspore/dvm/tree/master).

NPUGraph backend: Enabled via `torch.compile(backend="npugraphs")`, which leverages NPUGraphs technology to completely eliminate NPU task launch overhead and CPU-to-NPU synchronization overhead. It is suitable for scenarios where the eager mode is host-bound with frequent kernel calls but fixed input shapes. The overall functionality is consistent with `backend="cudagraphs"`. For details on NPUGraph's working principles, core advantages, API reference, and more usage examples, see [NPUGraph](pytorch_npugraph_desc.md).

NPUGraph_EX backend: Enabled via `torch.compile(backend="npugraph_ex")`, which accelerates large model inference based on ACLGraph scheduling and FX graph optimization, and integrates quickly and seamlessly with mainstream serving frameworks.

## Usage Guide

> [!NOTICE]
>
> The Inductor backend requires the latest version of the Triton-Ascend dependency package. For details, refer to the [Triton-Ascend documentation](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md).<br>
> When using the Inductor backend in MLIR mode, the Torch-MLIR dependency package must be additionally installed. It can be downloaded from the [Torch-MLIR archive](https://repo.oepkgs.net/ascend/pytorch/vllm/torch/).

Interface prototype:

```python
def compile(model, *, fullgraph = False, dynamic = None, backend = "inductor", mode = None, options = None, disable = False)
```

Parameter description:

- **model**: Required parameter, the model or parameter to be compiled.
- **fullgraph**: Optional parameter, whether to force full-graph compilation. Default value is False.
- **dynamic**: Optional parameter, whether dynamic shape compilation is needed. Default value is None.
- **backend**: optional parameter, the compilation backend, supporting inductor, npugraphs, and npugraph_ex. The default value is inductor.
- **mode**: optional parameter, the compilation mode. Currently supports None (default value) and "reduce-overhead".
- **options**: optional parameter, compilation options.

    - inductor and npugraphs support the following parameters:
      - triton.cudagraphs
      - trace.enabled
      - enable\_shape\_handling
      - npu\_backend
    - For parameters supported by npugraph_ex and detailed usage guidance, refer to the [npugraph_ex backend](https://gitcode.com/Ascend/torchair/blob/26.0.0/docs/en/npugraph_ex/npugraph_ex.md) in *PyTorch Graph Mode Usage (TorchAir)*.

- **disable**: optional parameter. Whether to disable the torch.compile capability. The default value is False.

For details about this API, refer to the native [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

## Usage Examples

- Inductor backend `torch.compile(backend="inductor")` example:

    ```Python
    import torch
    import torch_npu
    import torch.nn as nn
    
    # 1. Define a simple model (e.g., MLP)
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 10)
    
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    # 2. Compile the model (the core line)
    model = SimpleMLP().npu()
    compiled_model = torch.compile(
        model,
        backend="inductor",  # Specify the backend as Inductor
        mode="reduce-overhead"  # Optimization strategy: reduce overhead
    )
    # If you need to specify the operator compiler, add the option options={"npu_backend":"mlir"} or options={"npu_backend":"dvm"}

    # 3. Normal training/inference (usage is exactly the same as the original model)
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(32, 128).npu()  # Input tensor (batch_size=32)
    y = torch.randint(0, 10, (32,)).npu()
    

    for _ in range(100):  # Iterative training

        output = compiled_model(x)
        loss = criterion(output, y)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```

- NPUGraph backend `torch.compile(backend="npugraphs")` example:

    ```Python
    # Run
    import torch
    import torch.nn as nn
    
    # 1. Define the model (same as mode 1, ensure fixed input shape)
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 10)
    
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    # 2. Compile the model (specify the npugraphs backend)
    model = SimpleMLP().npu()
    compiled_model = torch.compile(
        model,
        backend="npugraphs"  # Core: enable NPUGraph optimization
    )
    
    # 3. Training/inference (must ensure fixed input shapes!)
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Note: npugraphs requires that each input shape, stride, etc. be exactly the same
    fixed_input = torch.randn(32, 128).npu()  # Fixed shape (32, 128)
    fixed_target = torch.randint(0, 10, (32,)).npu()
    
    for _ in range(1000):  # High iteration count scenario (benefits of repeated runs are more pronounced)
        output = compiled_model(fixed_input)
        loss = criterion(output, fixed_target)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```

- NPUGraph_EX backend `torch.compile(backend="npugraph_ex")` example:

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
        backend="npugraph_ex"  # Core: enable NPUGraph_EX optimization
    )
    
    # 3. Training/inference
    input_data = torch.randn(32, 128).npu() 
    
    for _ in range(1000):  # High iteration count scenarios (replay benefits are more significant)
        output = compiled_model(input_data)
    ```

## Constraints

1. The optimizer is typically not included in the graph. The optimizer's step\(\) involves Python-side dynamic logic (such as learning rate scheduling, gradient accumulation, and adaptive update rules), which is difficult to capture in graph mode.
2. torch.compile\(backend="npugraphs"\) requires fixed input shapes (batch\_size, sequence length, etc. cannot be modified after capture). torch.compile\(backend="inductor"\) supports dynamic shapes, but this triggers recompilation (incurring additional overhead). It is recommended to keep shapes fixed whenever possible.
3. When using NPUGraph (ACLGraph), determine whether operators need to be updated during the replay phase based on their characteristics. If updates are required, enable the corresponding update mechanism. For details on the mechanism, see [NPUGraph](pytorch_npugraph_desc.md).
4. NPUGraph (ACLGraph) can only be used when all operators are NN operators.
