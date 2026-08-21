# torch_npu.npu.NPUGraph

## Introduction

NPUGraph is a static graph capture technique used in Eager Mode (single-operator execution mode). It defines and encapsulates a series of NPU kernel definitions into a unit (that is, an operation graph), and starts multiple NPU operations through a single CPU operation, thereby reducing startup overhead. For more core principles, see [CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).

### Working Process

1. **Capture**: After the NPU stream is placed in capture mode, kernel calls are recorded as a computation graph structure.
2. **Replay**: After capture is complete, use `g.replay()` to replay the same computation logic multiple times. Each replay uses exactly the same kernel sequence and execution order as those during capture, without repeated graph construction or kernel launch preparation.

    Memory management during replay is the key point: the memory addresses of tensors allocated during the capture phase remain unchanged during replay. If you need to use new data between steps, you must write the new data to the memory addresses occupied during capture through `copy_()`. Do not directly reassign the tensor (such as `tensor = new_data`), because the variable would then point to new memory addresses and replay would still operate on the old addresses, causing data invalidation. This mechanism is automatically handled in the internal implementation of `make_graphed_callables`. For forward and backward propagation, when the `data_ptr()` of an input tensor is detected to differ from that during capture, `copy_()` is automatically executed to update it.

3. **Update**: For operators that require dynamic parameters (such as sequence length), for example FlashAttention, NPUGraph refreshes the parameter values before replay through a dedicated update mechanism, without recapturing the entire graph. This mechanism is automatically managed by the operator Handler in `_npugraph_handlers`, and you enable it through `auto_dispatch_capture=True`.

### Key Advantages

The graph replay mechanism trades the flexibility of dynamic graph execution for significantly reduced CPU overhead:

1. **Fixed computation graph structure**: Parameters and kernels are determined during the capture phase and no longer change. During replay, repeated parameter validation, kernel selection, and other operations are not required.
2. **Efficient execution process**: During replay, the underlying launch interface is directly invoked to submit the entire graph task to the NPU in batch, skipping the multi-layer scheduling overhead on the CPU side.

The following figure shows the advantages of NPUGraph:

![NPUGraph Advantages](../figures/npugraph.png)

> [!NOTE]
>
> When the CPU launches a series of short kernels one by one, the CPU launch overhead creates significant gaps between kernels. Using NPUGraph to replace this kernel sequence initially requires more time to build the graph and launch the entire graph at once, but subsequent executions are much faster because the gaps between kernels are very small. The difference becomes more obvious when the same operation sequence is repeated many times (for example, when the number of training steps is very large). The initial cost of building and launching the graph is amortized over the total number of training iterations.

## Applicable Scenarios

**Recommended scenarios for using NPUGraph:**

- The network structure is fully or partially a static graph (graph-safe)
- A CPU bottleneck exists (especially for short kernel-intensive tasks)
- Small-batch training scenarios (low NPU utilization)
- Inference or training tasks with fixed input shapes
- Repetitive computation tasks with a high number of iterations
- Only aclnn operators are supported for graph capture

**Scenarios where NPUGraph is not applicable:**

- Dynamic shape inputs (varying batch sizes)
- Dynamic control flow (variable conditional branches and loop structures)
- Operations that require frequent CPU-NPU synchronization

## Interface Description

NPUGraph provides three modes:

- NPUGraph class (`torch_npu.npu.NPUGraph`): Low-level control with manual management. It captures an entire continuous computation graph.
- graph context manager (`torch_npu.npu.graph`): Simplified capture. A simple and general-purpose context manager that captures NPU operations within its scope.
- make_graphed_callables (`torch_npu.npu.make_graphed_callables`): High-level encapsulation that handles details automatically. If parts of the network are not suitable for capture (for example, due to dynamic control flow, dynamic network topology, CPU synchronization, or critical CPU-side logic), you can use this high-level API. Consistent with the CUDA Graphs community, it automatically handles the graph capture details and the `copy_()` update of input data.

For a comparison of the applicable scenarios and operation modes of the three modes, see the following table.

| API | Manual operations | Automatic processing | Applicable scenarios |
|-----|--------|---------|---------|
| NPUGraph class (`capture_begin/end`) | Manually creates the Stream, calls `capture_begin/end`, manages the memory pool, and handles synchronization | ACLGraph capture and replay | Multi-Stream collaboration and region-based capture |
| graph() context manager | Determines the capture boundary | Automatically creates the Stream, and handles synchronization and cache | Quick start and single-stream scenarios | 
| `make_graphed_callables()` | Accepts callables and sample inputs | Automatically completes warmup, captures the forward and backward graphs, and handles data `copy_` updates | Safe subgraph capture with dynamic control flow | 

## Usage Examples

**Mode 1: Using the NPUGraph class (low-level API)**

`torch_npu.npu.NPUGraph` is the low-level primitive class that provides fine-grained control over the capture process. When using it, you must manually manage the Stream and call `capture_begin()` and `capture_end()`.

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

**Mode 2: Using the graph context manager**

Compared with manually calling `capture_begin()` and `capture_end()`, this mode is more concise and automatically handles Stream synchronization and cache cleanup.

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

**Mode 3: make_graphed_callables: Safe subgraph capture**

This API encapsulates the safe parts into graphed callable objects, while the remaining parts remain in eager execution.

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.optim as optim
from itertools import chain

def main():
    # 1. Check the environment: verify NPU availability
    if not torch.npu.is_available():
        print("# NPU is unavailable. Run this example in an NPU-supported environment")
        return
    print(f"# PyTorch version: {torch.__version__}")
    print(f"# torch_npu version: {torch_npu.__version__}")
    print(f"# Number of NPU devices: {torch.npu.device_count()}")

    # 2. Reproducibility settings
    torch.manual_seed(42)
    torch.npu.manual_seed(42)

    # 3. Define the model and migrate it to the NPU
    N, D_in, H, D_out = 640, 4096, 2048, 1024
    module1 = nn.Linear(D_in, H).npu()
    module2 = nn.Linear(H, D_out).npu()
    module3 = nn.Linear(H, D_out).npu()

    loss_fn = nn.MSELoss().npu()
    optimizer = optim.SGD(
        chain(module1.parameters(), module2.parameters(), module3.parameters()),
        lr=0.1
    )

    # 4. Prepare static tensors for capture (the requires_grad status must match the actual input)
    x = torch.randn(N, D_in, device='npu')  # module1 input does not require gradients
    h = torch.randn(N, H, device='npu', requires_grad=True)  # module2/3 inputs require gradients

    # 5. Use make_graphed_callables to capture subgraphs
    print("# Capturing NPUGraph subgraphs")
    module1 = torch_npu.npu.make_graphed_callables(module1, (x,))
    module2 = torch_npu.npu.make_graphed_callables(module2, (h,))
    module3 = torch_npu.npu.make_graphed_callables(module3, (h,))
    print("# NPUGraph subgraph capture completed")

    # 6. Prepare real training data
    real_inputs = [torch.randn_like(x) for _ in range(10)]
    real_targets = [torch.randn(N, D_out, device='npu') for _ in range(10)]

    # 7. Run training iterations (including dynamic branches)
    print("# Starting 10 iterations (using NPUGraphed Callables)")
    for i, (data, target) in enumerate(zip(real_inputs, real_targets)):
        optimizer.zero_grad(set_to_none=True)

        # Forward: module1 is executed unconditionally
        tmp = module1(data)  # graphed forward

        # Dynamic branch: select module2 or module3 based on the intermediate result
        # Note: NPUGraph requires the branch structure to be determined at capture time. The branch here only affects which graph is reused,
        #        and the computation graph inside each branch remains static, so it is safe to use
        if tmp.sum().item() > 0:
            tmp = module2(tmp)  # graphed forward
        else:
            tmp = module3(tmp)  # graphed forward

        loss = loss_fn(tmp, target)
        loss.backward()  # graphed backward for the selected module + module1 backward
        optimizer.step()

        if i == 0 or i == 9:
            param_sum = sum(p.sum().item() for p in chain(
                module1.parameters(), module2.parameters(), module3.parameters()))
            print(f"# Iteration {i+1}: total model parameter sum={param_sum:.6f}, loss={loss.item():.6f}")

    print("# All iterations completed")
    print("# NPUGraphed Callables verification succeeded")

if __name__ == "__main__":
    main()
```
