# Overview

## Introduction

TorchNPU has built a complete system for memory management, deeply integrating PyTorch's native memory management mechanisms while providing multiple unique optimization capabilities tailored to Ascend NPU hardware features. When using TorchNPU, developers can directly reuse most of PyTorch's native memory management interfaces and optimization methods, and can also enable NPU-specific memory optimization features as needed to cope with the growing memory challenges in large-model training.

The TorchNPU memory management system can be divided into the following three layers:

- **Native PyTorch Layer**: Inherits the memory management mechanisms of the PyTorch framework, including tensor lifecycle management, autograd memory optimization, gradient checkpoints, and mixed-precision training. These capabilities can be used directly on the NPU without additional adaptation.
- **NPU Device Memory Allocation Layer**: Centered on the NPU caching allocator, it provides interfaces and semantics consistent with the CUDA caching allocator, managing the allocation, caching, and recycling of NPU device memory.
- **NPU Feature Optimization Layer**: For the unique capabilities of Ascend hardware and large-model training scenarios, it provides differentiated features such as virtual memory, multistream memory reuse, and memory sharing to further reduce memory usage and memory fragmentation.

## TorchNPU Memory Management Features

The memory management features provided by TorchNPU itself are shown in the following table, covering scenarios such as memory allocation, memory optimization, memory monitoring, and inter-process sharing.

**Table 1**  TorchNPU memory management features

<table style="undefined;table-layout: fixed; width: 100%">
<colgroup>
<col style="width: 18%">
<col style="width: 32%">
<col style="width: 50%">
</colgroup>
<thead>
  <tr>
    <th>Feature Name</th>
    <th>Feature Description</th>
    <th>Applicable Scenario</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="./virtual_memory.md">Virtual Memory</a></td>
    <td>Through the expandable segment mechanism, virtual addresses are separated from physical memory, allowing multiple allocations of contiguous memory and dynamic adjustment of memory block sizes, effectively reducing memory fragmentation.</td>
    <td>When memory fragmentation frequently causes OOM during training, or when the model memory usage is high.</td>
  </tr>
  <tr>
    <td><a href="./memory_snapshot.md">Memory Snapshot</a></td>
    <td>Generates device memory snapshots proactively on OOM or through APIs, recording memory allocation states and history, and supports visual analysis through memory_viz.</td>
    <td>When you need to analyze NPU memory allocation and troubleshoot OOM causes.</td>
  </tr>
  <tr>
    <td><a href="./custom_memory_allocator.md">Custom Memory Allocator</a></td>
    <td>Supports loading user-defined NPU memory allocators from .so files to replace the default caching allocator.</td>
    <td>Scenarios with special memory management requirements, such as the need for custom memory allocation strategies.</td>
  </tr>
  <tr>
    <td><a href="./multistream_memory_reuse.md">Multi-Stream Memory Reuse</a></td>
    <td>Optimizes memory usage in multistream scenarios, improving memory utilization through a cross-stream reuse mechanism.</td>
    <td>Training or inference scenarios with multistream parallel execution.</td>
  </tr>
  <tr>
    <td><a href="./memory_sharing_ipc.md">Memory Sharing (IPC)</a></td>
    <td>Supports sharing NPU memory across processes, passing tensor data between different processes through the IPC mechanism to reduce overall memory consumption.</td>
    <td>Multi-process data sharing scenarios, such as data transfer between the data loading process and the training process.</td>
  </tr>
  <tr>
    <td><a href="../operator_dispatch/taskqueue/stream_taskqueue_parallel_delivery.md">Stream-Level TaskQueue Parallel Delivery</a></td>
    <td>Each Stream initializes an independent TaskQueue and Dequeue thread, implementing a two-level pipelined parallel delivery mechanism that improves compute efficiency while optimizing memory usage.</td>
    <td>Scenarios that require improved delivery performance and full use of multistream parallelism.</td>
  </tr>
</tbody>
</table>

## Reusing Native PyTorch Memory Features

Because TorchNPU inherits PyTorch's programming model and API system, PyTorch's native memory-related basic features can be used directly on the NPU without modifying code. The following table lists common native PyTorch memory features and their usage instructions on the NPU.

**Table 2**  Native PyTorch memory features that can be directly reused on the NPU

<table style="undefined;table-layout: fixed; width: 100%">
<colgroup>
<col style="width: 10%">
<col style="width: 15%">
<col style="width: 24%">
<col style="width: 28%">
<col style="width: 23%">
</colgroup>
<thead>
  <tr>
    <th>Feature Category</th>
    <th>Feature Name</th>
    <th>PyTorch API/Method</th>
    <th>NPU Usage Instructions</th>
    <th>PyTorch Upstream Documentation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3">Memory allocation</td>
    <td>Tensor creation</td>
    <td><code>torch.empty()</code>, <code>torch.zeros()</code>, <code>torch.ones()</code>, <code>torch.rand()</code>, etc.</td>
    <td>Specify <code>device='npu'</code> to allocate memory on the NPU. The interfaces and semantics are exactly consistent with CUDA.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.empty.html">torch.empty</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.zeros.html">torch.zeros</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.ones.html">torch.ones</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.rand.html">torch.rand</a></td>
  </tr>
  <tr>
    <td>Tensor type conversion</td>
    <td><code>tensor.to()</code>, <code>tensor.cuda()</code>, <code>tensor.npu()</code>, <code>tensor.half()</code>, <code>tensor.bfloat16()</code>, etc.</td>
    <td>Supports conversion from CPU to NPU, from NPU to CPU, and between different dtypes on the NPU. <code>tensor.npu()</code> is NPU-specific syntactic sugar.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html">torch.Tensor.to</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cuda.html">torch.Tensor.cuda</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.half.html">torch.Tensor.half</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.bfloat16.html">torch.Tensor.bfloat16</a><br><code>tensor.npu()</code> is an NPU-specific interface, corresponding to <code>Tensor.cuda()</code> upstream.</td>
  </tr>
  <tr>
    <td>Pin Memory</td>
    <td><code>tensor.pin_memory()</code>, <code>DataLoader(pin_memory=True)</code></td>
    <td>The pinned memory feature is also supported on the NPU and can accelerate data transfer from CPU to NPU.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html">torch.Tensor.pin_memory</a><br><a href="https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader">torch.utils.data.DataLoader</a><br><a href="https://docs.pytorch.org/docs/stable/data.html#memory-pinning">DataLoader / Memory Pinning</a></td>
  </tr>
  <tr>
    <td rowspan="7">Memory optimization</td>
    <td>View operations</td>
    <td><code>tensor.view()</code>, <code>tensor.reshape()</code>, <code>tensor.permute()</code>, <code>tensor.transpose()</code>, etc.</td>
    <td>View operations do not allocate new memory and share storage with the original tensor. They can be used directly on the NPU.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html">torch.Tensor.view</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html">torch.Tensor.reshape</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.permute.html">torch.Tensor.permute</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.transpose.html">torch.Tensor.transpose</a></td>
  </tr>
  <tr>
    <td>In-place operations</td>
    <td><code>tensor.add_()</code>, <code>tensor.mul_()</code>, <code>tensor.relu_()</code>, etc. (operators with an underscore suffix)</td>
    <td>In-place operations reuse existing memory to avoid additional allocation. All operators supporting in-place operations on the NPU behave consistently with CUDA.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.add_.html">torch.Tensor.add_</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.mul_.html">torch.Tensor.mul_</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu_.html">torch.nn.functional.relu_ (corresponding to tensor.relu_)</a></td>
  </tr>
  <tr>
    <td>Inference mode</td>
    <td><code>torch.no_grad()</code>, <code>torch.inference_mode()</code></td>
    <td>Disables autograd, avoiding saving intermediate activations for backpropagation and significantly reducing memory usage. The behavior on the NPU is exactly consistent with CUDA.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html">torch.no_grad</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html">torch.inference_mode</a></td>
  </tr>
  <tr>
    <td>Mixed-precision training (AMP)</td>
    <td><code>torch.amp.autocast(device_type='npu')</code>, <code>torch.npu.amp.autocast()</code></td>
    <td>Reduces memory usage and computation through FP16/BF16. The NPU supports the same AMP interfaces as CUDA. Specify <code>device_type='npu'</code> or use NPU-specific interfaces.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/amp.html#torch.autocast">torch.amp.autocast</a><br><a href="https://docs.pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast">torch.cuda.amp.autocast</a></td>
  </tr>
  <tr>
    <td>Gradient checkpointing</td>
    <td><code>torch.utils.checkpoint.checkpoint()</code></td>
    <td>Trades computation for memory: intermediate activations are not saved during the forward pass and are recomputed during the backward pass. It can be used directly on the NPU, with the same usage as CUDA.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint">torch.utils.checkpoint.checkpoint</a></td>
  </tr>
  <tr>
    <td>Parameter sharing</td>
    <td>Assigning the same tensor to parameters of multiple modules</td>
    <td>Multiple modules share the same weight memory, reducing the memory usage corresponding to the model parameter count. The behavior on the NPU is consistent with CUDA.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/notes/modules.html#module-state">Module State</a></td>
  </tr>
  <tr>
    <td>Model checkpoint management</td>
    <td><code>torch.save()</code>, <code>torch.load()</code></td>
    <td>When saving/loading models, control the tensor loading location through <code>map_location='npu'</code> or <code>map_location='cpu'</code> to flexibly manage device memory.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.save.html">torch.save</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.load.html">torch.load</a></td>
  </tr>
  <tr>
    <td rowspan="4">Memory monitoring</td>
    <td>Memory statistics</td>
    <td><code>torch.npu.memory_allocated()</code>, <code>torch.npu.max_memory_allocated()</code>, <code>torch.npu.memory_reserved()</code>, <code>torch.npu.max_memory_reserved()</code></td>
    <td>The NPU provides interfaces fully corresponding to <code>torch.cuda.memory_*()</code>. The usage is the same, returning the memory usage statistics of the current device.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.memory_allocated.html">memory_allocated</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html">max_memory_allocated</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.memory_reserved.html">memory_reserved</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html">max_memory_reserved</a></td>
  </tr>
  <tr>
    <td>Memory overview</td>
    <td><code>torch.npu.memory_summary()</code></td>
    <td>Returns a formatted summary report of memory usage, corresponding to <code>torch.cuda.memory_summary()</code>.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.memory_summary.html">memory_summary</a></td>
  </tr>
  <tr>
    <td>Cache clearing</td>
    <td><code>torch.npu.empty_cache()</code></td>
    <td>Releases unused cached memory in the caching allocator, corresponding to <code>torch.cuda.empty_cache()</code>.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.empty_cache.html">empty_cache</a></td>
  </tr>
  <tr>
    <td>Memory peak reset</td>
    <td><code>torch.npu.reset_peak_memory_stats()</code>, <code>torch.npu.reset_accumulated_memory_stats()</code></td>
    <td>Resets the memory statistics counters, corresponding to interfaces such as <code>torch.cuda.reset_peak_memory_stats()</code>.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.reset_peak_memory_stats.html">reset_peak_memory_stats</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.reset_accumulated_memory_stats.html">reset_accumulated_memory_stats</a></td>
  </tr>
  <tr>
    <td rowspan="3">Memory management</td>
    <td>Memory pool (MemPool)</td>
    <td><code>torch.npu.MemPool()</code>, <code>torch.npu.use_mem_pool()</code></td>
    <td>Supports users in creating and using independent memory pools, routing the memory allocation of specific tensors to a designated memory pool. Corresponds to <code>torch.cuda.MemPool()</code>.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.MemPool.html">MemPool</a><br><a href="https://docs.pytorch.org/docs/stable/cuda.html#torch.cuda.use_mem_pool">torch.cuda.use_mem_pool</a></td>
  </tr>
  <tr>
    <td>Per-process memory limit</td>
    <td><code>torch.npu.set_per_process_memory_fraction()</code>, <code>torch.npu.get_per_process_memory_fraction()</code></td>
    <td>Sets/gets the maximum NPU memory fraction that the current process can occupy. Corresponds to <code>torch.cuda.set_per_process_memory_fraction()</code>.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.set_per_process_memory_fraction.html">set_per_process_memory_fraction</a><br><a href="https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.get_per_process_memory_fraction.html">get_per_process_memory_fraction</a></td>
  </tr>
  <tr>
    <td>Memory snapshot API</td>
    <td><code>torch.npu.memory._record_memory_history()</code>, <code>torch.npu.memory._dump_snapshot()</code>, <code>torch.npu.memory._snapshot()</code></td>
    <td>Proactively records memory allocation history and exports snapshots for analysis. Corresponds to interfaces such as <code>torch.cuda.memory._record_memory_history()</code>.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._record_memory_history">_record_memory_history</a><br><a href="https://docs.pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._dump_snapshot">_dump_snapshot</a><br><a href="https://docs.pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._snapshot">_snapshot</a></td>
  </tr>
  <tr>
    <td rowspan="2">Debugging and analysis</td>
    <td>Memory visualization</td>
    <td><code>torch.npu.memory._save_segment_usage()</code>, <code>torch.npu.memory._save_memory_usage()</code></td>
    <td>Generates an SVG flame graph of memory usage to intuitively show the memory allocation layout. Corresponds to the CUDA-side <code>_memory_viz</code> tool.</td>
    <td><a href="https://github.com/pytorch/pytorch/blob/main/torch/cuda/_memory_viz.py">torch.cuda._memory_viz.segments / memory (SVG visualization source code)</a></td>
  </tr>
  <tr>
    <td>PyTorch Profiler</td>
    <td><code>torch.profiler.profile()</code> + <code>profile_memory=True</code></td>
    <td>Collects the NPU memory allocation timeline and peak information through the PyTorch Profiler for visual analysis in TensorBoard.</td>
    <td><a href="https://docs.pytorch.org/docs/stable/profiler.html#torch.profiler.profile">torch.profiler.profile</a></td>
  </tr>
</tbody>
</table>

> [!NOTE]
> The `torch.npu.*` interfaces listed in the above table are NPU-specific memory management APIs provided by TorchNPU. Their interface signatures and semantics are consistent with the corresponding `torch.cuda.*` interfaces. If your code was previously written based on CUDA, you usually only need to replace `cuda` with `npu` to complete the migration.

## Related References

- For environment variable configuration, see the "[Memory Management](../../api/environment_variable/memory_management/_menu_memory_management.md)" section in Environment Variable Reference.
- For detailed usage instructions of each feature, see the subsections of this section.
- For more NPU memory APIs, see the [PyTorch CUDA Memory Management Documentation](https://pytorch.org/docs/stable/torch_cuda_memory.html) (NPU interfaces correspond one-to-one to CUDA interfaces).
