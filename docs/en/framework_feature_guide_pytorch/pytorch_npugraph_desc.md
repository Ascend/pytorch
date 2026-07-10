# NPUGraph

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-08T10:21:33.595Z pushedAt=2026-07-08T10:47:16.860Z -->

## Introduction

NPUGraph is a static graph capture technique that converts dynamic PyTorch operations into a fixed computation graph, improving NPU execution efficiency. This technique defines and encapsulates a series of NPU kernels as a single unit (i.e., an operation graph), rather than a sequence of individually launched operations. It provides a mechanism to launch multiple NPU operations through a single CPU operation, thereby reducing launch overhead. Its core idea is consistent with [CUDAGraphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).

### How It Works

PyTorch supports building NPU computation graphs through a stream capture mechanism. The specific process is as follows:

1. NPU graph capture mode<br>
Start capture: After placing the NPU stream in capture mode, NPU work sent to that stream is not executed immediately; instead, it is recorded as a computation graph structure.<br>
Graph recording: During capture, all kernel calls and their parameters (including pointer addresses) are statically recorded.
2. Graph replay and memory management<br>
Replay mechanism: After capture is complete, the same computation logic can be replayed multiple times by launching the graph, with each replay using the exact same kernels and parameters.<br>
Dynamic data update: For pointer parameters (such as input tensors), new data (e.g., a new batch) must be written to the original memory addresses before each replay, enabling data updates without rebuilding the graph.

### Key advantages

The graph replay mechanism trades the flexibility of dynamic graph execution for significantly reduced CPU overhead. The specific advantages are as follows:

1. Fixed computation graph structure<br>
Parameters and kernels are determined during the capture phase and remain unchanged thereafter, eliminating the need for repeated parameter validation, kernel selection, and other operations during replay. This avoids scheduling overhead from the Python interpreter, C++ framework layer, and driver layer.
2. Efficient execution flow<br>
During replay, the underlying launch interface is invoked directly, submitting the entire graph task to the NPU in batch. NPU kernel execution efficiency sees a modest improvement, but the core advantage lies in bypassing multi-layer CPU scheduling overhead (such as kernel launch preparation, memory mapping management, etc.).

The advantages of NPUGraph can be illustrated by the following diagram:

![NPUGraph Advantages](../figures/npugraph.png)
> [!NOTE]
>
> When the CPU launches a series of short kernels one by one, the CPU launch overhead creates significant gaps between kernels. In contrast, using NPUGraph to replace this kernel sequence initially requires more time to construct the graph and launch the entire graph at once, but subsequent executions are much faster because the gaps between kernels are very small. The difference becomes more pronounced when the same operation sequence is repeated many times (for example, with a large number of training steps). The initial cost of constructing and launching the graph is amortized over the entire training iterations.

## Applicable Scenarios

**Recommended scenarios for using NPUGraph:**

- The network structure is fully or partially a static graph (graph-safe)
- CPU bottleneck exists (especially for short kernel-intensive tasks)
- Small-batch training scenarios (low NPU utilization)
- Inference or training tasks with fixed input shapes
- Repetitive computation tasks with a high number of iterations

**NPUGraph usage restrictions and considerations:**

- Currently, only aclnn operators are supported for graph capture
- Not applicable to dynamic shape inputs (batch sizes that vary across iterations)
- Not applicable to dynamic control flow (variable conditional branches or loop structures)
- Not applicable to operations that require frequent CPU-NPU synchronization

## API Overview

NPUGraph provides three usage modes:

- NPUGraph class (`torch_npu.npu.NPUGraph`): Low-level control with manual management.
- graph context manager (`torch_npu.npu.graph`): Simplified capture. A simple and general-purpose context manager that captures NPU operations within its scope.
- make_graphed_callables (`torch_npu.npu.make_graphed_callables`): High-level encapsulation that handles details automatically. If parts of the network are unsuitable for capture (for example, due to dynamic control flow, dynamic network topology, CPU synchronization, or critical CPU-side logic), this API can be used for safe subgraph capture.

## Usage Examples

### Enable Mode 1: Using the NPUGraph Class

`torch_npu.npu.NPUGraph` is the low-level primitive class that provides fine-grained control over the capture process. It requires manual stream management and explicit calls to `capture_begin()` and `capture_end()`.

```python
import torch
import torch_npu

def graph_capture_simple():
    s = torch_npu.npu.Stream()

    with torch_npu.npu.stream(s):
        a = torch.full((1000,), 1, device="npu")
        g = torch_npu.npu.NPUGraph()
        torch_npu.npu.empty_cache()
        g.capture_begin()
        b = a
        for _ in range(10):
            b = b + 1
        g.capture_end()
    torch_npu.npu.current_stream().wait_stream(s)

    g.replay()

    print(f"b.sum().item() == {b.sum().item()}.")

graph_capture_simple()
```

### Enable Mode 2: Using the graph Context Manager

`torch_npu.npu.graph` is a simple, general-purpose context manager that captures NPU operations within its scope. Compared to manually calling `capture_begin()` and `capture_end()`, this approach is more concise and automatically handles stream synchronization and cache cleanup.

```python
import torch
import torch_npu

def graph_simple():
    a = torch.full((1000,), 1, device="npu")
    g = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(g):
        b = a
        for _ in range(10):
            b = b + 1

    g.replay()

    print(f"b.sum().item() == {b.sum().item()}.")

graph_simple()
```

### Enable Mode 3: Safe Subgraph Capture

When the network contains non-capturable parts (such as dynamic control flow, dynamic shapes, CPU synchronization, or necessary CPU-side logic), the safe parts can be wrapped as graphed callables via `torch_npu.npu.make_graphed_callables`, while the remaining parts continue to execute in eager mode.

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.optim as optim
from itertools import chain

def main():
    # 1. Environment check: verify NPU availability
    if not torch.npu.is_available():
        print("# NPU is unavailable. Please run this example in an environment that supports NPU")
        return
    print(f"# PyTorch version: {torch.__version__}")
    print(f"# torch_npu version: {torch_npu.__version__}")
    print(f"# NPU device count: {torch.npu.device_count()}")

    # 2. Reproducibility setup
    torch.manual_seed(42)
    torch.npu.manual_seed(42)

    # 3. Model definition and NPU migration
    N, D_in, H, D_out = 640, 4096, 2048, 1024
    module1 = nn.Linear(D_in, H).npu()
    module2 = nn.Linear(H, D_out).npu()
    module3 = nn.Linear(H, D_out).npu()

    loss_fn = nn.MSELoss().npu()
    optimizer = optim.SGD(
        chain(module1.parameters(), module2.parameters(), module3.parameters()),
        lr=0.1
    )

    # 4. Prepare static tensors for capture (requires_grad status must match actual inputs)
    x = torch.randn(N, D_in, device='npu')  # module1 input does not require gradients
    h = torch.randn(N, H, device='npu', requires_grad=True)  # module2/3 input requires gradient

    # 5. Use make_graphed_callables to capture subgraphs
    print("# Capture NPUGraph subgraphs")
    module1 = torch_npu.npu.make_graphed_callables(module1, (x,))
    module2 = torch_npu.npu.make_graphed_callables(module2, (h,))
    module3 = torch_npu.npu.make_graphed_callables(module3, (h,))
    print("# NPUGraph subgraph capture completed")

    # 6. Prepare real training data
    real_inputs = [torch.randn_like(x) for _ in range(10)]
    real_targets = [torch.randn(N, D_out, device='npu') for _ in range(10)]

    # 7. Execute training iterations (with dynamic branching)
    print("# Start 10 iterations (using NPUGraphed callables)")
    for i, (data, target) in enumerate(zip(real_inputs, real_targets)):
        optimizer.zero_grad(set_to_none=True)

        # Forward: module1 executes unconditionally
        tmp = module1(data)  # Graphed forward

        # Dynamic branch: select module2 or module3 based on intermediate results
        # ⚠️ Note: NPUGraph requires the branch structure to be determined at capture time; the branch here only affects which graph is reused,
        #        and the computation graph within each branch remains static, so it is safe to use
        if tmp.sum().item() > 0:
            tmp = module2(tmp)  # Graphed forward
        else:
            tmp = module3(tmp)  # Graphed forward

        loss = loss_fn(tmp, target)
        loss.backward()  # Graphed backward for the selected module + module1 backward
        optimizer.step()

        if i == 0 or i == 9:
            param_sum = sum(p.sum().item() for p in chain(
                module1.parameters(), module2.parameters(), module3.parameters()))
            print(f"# Iteration {i+1}: total model params={param_sum:.6f}, loss={loss.item():.6f}")

    print("# All iterations completed")
    print("# NPUGraphed callables functionality verified successfully")

if __name__ == "__main__":
    main()
```
