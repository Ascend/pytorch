# 内存资源优化概述

## 简介

TorchNPU在内存管理方面构建了一套完整的体系，既深度集成了PyTorch原生的内存管理机制，又针对昇腾NPU硬件特性提供了多项独有的优化能力。开发者在使用TorchNPU时，大部分PyTorch原生的内存管理接口和优化手段可直接复用，同时也可按需开启NPU特有的内存优化特性，以应对大模型训练中日益增长的内存挑战。

TorchNPU的内存管理体系从层次上可分为以下三层：

- **PyTorch原生层**：继承PyTorch框架的内存管理机制，包括张量生命周期管理、自动求导内存优化、梯度检查点、混合精度训练等，这些能力在NPU上可直接使用，无需额外适配。
- **NPU设备内存分配层**：以NPU缓存分配器（NPU Caching Allocator）为核心，提供与CUDA缓存分配器一致的接口和语义，管理NPU设备内存的分配、缓存与回收。
- **NPU特性优化层**：针对昇腾硬件的特有能力和大模型训练场景，提供虚拟内存、多流内存复用、内存共享等差异化特性，进一步降低内存占用、减少内存碎片。

## TorchNPU内存管理功能

TorchNPU自身提供的内存管理功能如下表所示，涵盖内存分配、内存优化、内存监控和进程间共享等场景。

**表 1**  TorchNPU内存管理功能

<table style="undefined;table-layout: fixed; width: 100%">
<colgroup>
<col style="width: 18%">
<col style="width: 32%">
<col style="width: 50%">
</colgroup>
<thead>
  <tr>
    <th>功能名称</th>
    <th>功能说明</th>
    <th>适用场景</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="./memory_resource_optimization/virtual_memory.md">虚拟内存</a></td>
    <td>通过可扩展内存段机制，将虚拟地址与物理内存分离，允许多次申请连续内存并动态调整内存块大小，有效减少内存碎片。</td>
    <td>训练过程中频繁出现内存碎片导致OOM，或模型内存占用率高时。</td>
  </tr>
  <tr>
    <td><a href="./memory_resource_optimization/memory_snapshot.md">内存快照</a></td>
    <td>在OOM时或通过API主动生成设备内存快照，记录内存分配状态和历史记录，支持通过memory_viz进行可视化分析。</td>
    <td>需要分析NPU内存分配情况、排查OOM原因时。</td>
  </tr>
  <tr>
    <td><a href="./custom_memory_allocator.md">自定义内存分配器</a></td>
    <td>支持从.so文件加载用户自定义的NPU内存分配器，替换默认的缓存分配器。</td>
    <td>有特殊内存管理需求的场景，如需要自定义内存分配策略。</td>
  </tr>
  <tr>
    <td><a href="./multistream_memory_reuse.md">多流内存复用</a></td>
    <td>在多流场景下优化内存使用，通过跨流复用机制提高内存利用率。</td>
    <td>多流并行执行的训练或推理场景。</td>
  </tr>
  <tr>
    <td><a href="./memory_sharing_ipc.md">内存共享（IPC）</a></td>
    <td>支持跨进程共享NPU内存，通过IPC机制在不同进程间传递张量数据，减少整体内存消耗。</td>
    <td>多进程数据共享场景，如数据加载进程与训练进程间的数据传输。</td>
  </tr>
  <tr>
    <td><a href="../operator_dispatch/stream_taskqueue_parallel_delivery.md">Stream级TaskQueue并行下发</a></td>
    <td>每个Stream初始化独立的TaskQueue和Dequeue线程，实现二级流水并行下发机制，提升计算效率的同时优化内存使用。</td>
    <td>需要提升下发性能、充分利用多Stream并行的场景。</td>
  </tr>
</tbody>
</table>

## 原生PyTorch内存功能复用

由于TorchNPU继承了PyTorch的编程模型和API体系，PyTorch原生的内存相关基础功能可直接在NPU上使用，无需修改代码。下表列出了常用的原生PyTorch内存功能及其在NPU上的使用说明。

**表 2**  可在NPU上直接复用的原生PyTorch内存功能

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
    <th>功能类别</th>
    <th>功能名称</th>
    <th>PyTorch接口/方式</th>
    <th>NPU使用说明</th>
    <th>PyTorch上游文档</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3">内存分配</td>
    <td>张量创建</td>
    <td><code>torch.empty()</code>、<code>torch.zeros()</code>、<code>torch.ones()</code>、<code>torch.rand()</code> 等</td>
    <td>指定 <code>device='npu'</code> 即可在NPU上分配内存，接口和语义与CUDA完全一致。</td>
    <td><a href="https://pytorch.org/docs/stable/tensors.html">torch.Tensor</a></td>
  </tr>
  <tr>
    <td>张量类型转换</td>
    <td><code>tensor.to()</code>、<code>tensor.cuda()</code>、<code>tensor.npu()</code>、<code>tensor.half()</code>、<code>tensor.bfloat16()</code> 等</td>
    <td>支持CPU到NPU、NPU到CPU以及NPU上不同dtype之间的转换。<code>tensor.npu()</code> 是NPU专用语法糖。</td>
    <td><a href="https://pytorch.org/docs/stable/tensors.html">torch.Tensor</a></td>
  </tr>
  <tr>
    <td>Pin Memory</td>
    <td><code>tensor.pin_memory()</code>、<code>DataLoader(pin_memory=True)</code></td>
    <td>锁页内存功能在NPU上同样支持，可加速CPU到NPU的数据传输。</td>
    <td><a href="https://pytorch.org/docs/stable/data.html#memory-pinning">torch.utils.data</a></td>
  </tr>
  <tr>
    <td rowspan="7">内存优化</td>
    <td>视图操作</td>
    <td><code>tensor.view()</code>、<code>tensor.reshape()</code>、<code>tensor.permute()</code>、<code>tensor.transpose()</code> 等</td>
    <td>视图操作不分配新内存，与原张量共享存储，可直接在NPU上使用。</td>
    <td><a href="https://pytorch.org/docs/stable/tensor_view.html">Tensor Views</a></td>
  </tr>
  <tr>
    <td>原位操作</td>
    <td><code>tensor.add_()</code>、<code>tensor.mul_()</code>、<code>tensor.relu_()</code> 等（带下划线后缀的算子）</td>
    <td>原位操作复用已有内存，避免额外分配。NPU上所有支持原位操作的算子与CUDA行为一致。</td>
    <td><a href="https://pytorch.org/docs/stable/tensors.html">torch.Tensor</a></td>
  </tr>
  <tr>
    <td>推理模式</td>
    <td><code>torch.no_grad()</code>、<code>torch.inference_mode()</code></td>
    <td>禁用自动求导，避免为反向传播保存中间激活，显著降低内存占用。NPU上行为与CUDA完全一致。</td>
    <td><a href="https://pytorch.org/docs/stable/notes/autograd.html">Autograd</a></td>
  </tr>
  <tr>
    <td>混合精度训练（AMP）</td>
    <td><code>torch.amp.autocast(device_type='npu')</code>、<code>torch.npu.amp.autocast()</code></td>
    <td>通过FP16/BF16降低内存占用和计算量。NPU支持与CUDA相同的AMP接口，需指定 <code>device_type='npu'</code> 或使用NPU专用接口。</td>
    <td><a href="https://pytorch.org/docs/stable/notes/amp_examples.html">Automatic Mixed Precision</a></td>
  </tr>
  <tr>
    <td>梯度检查点</td>
    <td><code>torch.utils.checkpoint.checkpoint()</code></td>
    <td>以计算换内存，在前向过程中不保存中间激活，反向时重新计算。NPU上可直接使用，与CUDA用法相同。</td>
    <td><a href="https://pytorch.org/docs/stable/checkpoint.html">Checkpointing</a></td>
  </tr>
  <tr>
    <td>参数共享</td>
    <td>将同一张量赋值给多个模块参数</td>
    <td>多个模块共享同一份权重内存，减少模型参数量对应的内存占用。NPU上行为与CUDA一致。</td>
    <td><a href="https://pytorch.org/docs/stable/notes/modules.html#module-state">Module State</a></td>
  </tr>
  <tr>
    <td>模型Checkpoint管理</td>
    <td><code>torch.save()</code>、<code>torch.load()</code></td>
    <td>保存/加载模型时，可通过 <code>map_location='npu'</code> 或 <code>map_location='cpu'</code> 控制张量加载位置，灵活管理设备内存。</td>
    <td><a href="https://pytorch.org/docs/stable/notes/serialization.html">Serialization</a></td>
  </tr>
  <tr>
    <td rowspan="4">内存监控</td>
    <td>内存统计</td>
    <td><code>torch.npu.memory_allocated()</code>、<code>torch.npu.max_memory_allocated()</code>、<code>torch.npu.memory_reserved()</code>、<code>torch.npu.max_memory_reserved()</code></td>
    <td>NPU提供与 <code>torch.cuda.memory_*()</code> 完全对应的接口，用法相同，返回当前设备的内存使用统计。</td>
    <td><a href="https://pytorch.org/docs/stable/notes/cuda.html">CUDA Memory Management</a></td>
  </tr>
  <tr>
    <td>内存概览</td>
    <td><code>torch.npu.memory_summary()</code></td>
    <td>返回格式化的内存使用摘要报告，与 <code>torch.cuda.memory_summary()</code> 对应。</td>
    <td><a href="https://pytorch.org/docs/stable/notes/cuda.html">CUDA Memory Management</a></td>
  </tr>
  <tr>
    <td>缓存清理</td>
    <td><code>torch.npu.empty_cache()</code></td>
    <td>释放缓存分配器中未使用的缓存内存，与 <code>torch.cuda.empty_cache()</code> 对应。</td>
    <td><a href="https://pytorch.org/docs/stable/notes/cuda.html">CUDA Memory Management</a></td>
  </tr>
  <tr>
    <td>内存峰值重置</td>
    <td><code>torch.npu.reset_peak_memory_stats()</code>、<code>torch.npu.reset_accumulated_memory_stats()</code></td>
    <td>重置内存统计计数器，与 <code>torch.cuda.reset_peak_memory_stats()</code> 等接口对应。</td>
    <td><a href="https://pytorch.org/docs/stable/notes/cuda.html">CUDA Memory Management</a></td>
  </tr>
  <tr>
    <td rowspan="3">内存管理</td>
    <td>内存池（MemPool）</td>
    <td><code>torch.npu.MemPool()</code>、<code>torch.npu.use_mem_pool()</code></td>
    <td>支持用户创建和使用独立的内存池，将特定张量的内存分配路由到指定内存池。与 <code>torch.cuda.MemPool()</code> 对应。</td>
    <td><a href="https://pytorch.org/docs/stable/notes/cuda.html">CUDA Memory Management</a></td>
  </tr>
  <tr>
    <td>进程内存限制</td>
    <td><code>torch.npu.set_per_process_memory_fraction()</code>、<code>torch.npu.get_per_process_memory_fraction()</code></td>
    <td>设置/获取当前进程可占用的最大NPU内存比例。与 <code>torch.cuda.set_per_process_memory_fraction()</code> 对应。</td>
    <td><a href="https://pytorch.org/docs/stable/notes/cuda.html">CUDA Memory Management</a></td>
  </tr>
  <tr>
    <td>内存快照API</td>
    <td><code>torch.npu.memory._record_memory_history()</code>、<code>torch.npu.memory._dump_snapshot()</code>、<code>torch.npu.memory._snapshot()</code></td>
    <td>主动记录内存分配历史并导出快照进行分析。与 <code>torch.cuda.memory._record_memory_history()</code> 等接口对应。</td>
    <td><a href="https://pytorch.org/docs/stable/notes/cuda.html">CUDA Memory Management</a></td>
  </tr>
  <tr>
    <td rowspan="2">调试分析</td>
    <td>内存可视化</td>
    <td><code>torch.npu.memory._save_segment_usage()</code>、<code>torch.npu.memory._save_memory_usage()</code></td>
    <td>生成内存使用的SVG火焰图，直观展示内存分配布局。与CUDA端 <code>_memory_viz</code> 工具对应。</td>
    <td><a href="https://pytorch.org/docs/stable/notes/cuda.html">CUDA Memory Management</a></td>
  </tr>
  <tr>
    <td>PyTorch Profiler</td>
    <td><code>torch.profiler.profile()</code> + <code>profile_memory=True</code></td>
    <td>通过PyTorch Profiler采集NPU内存分配时间线和峰值信息，在TensorBoard中可视化分析。</td>
    <td><a href="https://pytorch.org/docs/stable/profiler.html">torch.profiler</a></td>
  </tr>
</tbody>
</table>

> [!NOTE]
> 上表中列出的 `torch.npu.*` 接口是TorchNPU提供的NPU专用内存管理API，其接口签名和语义与对应的 `torch.cuda.*` 接口保持一致。如果您的代码之前基于CUDA编写，通常只需将 `cuda` 替换为 `npu` 即可完成迁移。

## 相关参考

- 环境变量配置请参考《环境变量参考》中的[内存管理](../../api/environment_variable/memory_management/_menu_memory_management.md)章节。
- 各特性的详细使用说明请参考本节的子章节。
- 更多NPU内存API请参考 [PyTorch CUDA内存管理文档](https://pytorch.org/docs/stable/torch_cuda_memory.html)（NPU接口与CUDA接口一一对应）。
