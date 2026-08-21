# Overview

TorchNPU is a deep learning adaptation framework based on Ascend. It enables Ascend NPU to support the PyTorch framework and provides PyTorch framework users with the exceptional computing power of Ascend AI processors.

This plugin inherits the dynamic graph features, native development paradigm, and architecture of the PyTorch framework to the greatest extent possible, so that you can use TorchNPU with almost no changes to your original development habits or coding style. You can continue using familiar PyTorch interfaces and functions. As long as you specify Ascend NPU as the device, you can seamlessly migrate models to the Ascend platform for training, greatly reducing development costs and migration difficulty.

As deep learning flourishes, data scales keep expanding, and model complexity continues to climb, the challenges to computing platforms are growing. To address these challenges, TorchNPU has carefully built a series of unique features in areas such as memory resource optimization, communication performance optimization, computing performance optimization, and assisted error location, providing users with an efficient and convenient development tool. For the specific features, see [Table 1](#featurelist).

**Table 1**  Feature list <a id="featurelist"></a>
<table style="undefined;table-layout: fixed; width: 1508px"><colgroup>
<col style="width: 293px">
<col style="width: 252px">
<col style="width: 963px">
</colgroup>
<thead>
  <tr>
    <th>Feature category<br></th>
    <th>Feature name<br></th>
    <th>Feature description<br></th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5">Memory Resource Optimization</td>
    <td><a href="./virtual_memory.md">Virtual Memory</a></td>
    <td>Dynamically adjusts the memory block size to reduce memory fragmentation.</td>
  </tr>
  <tr>
    <td><a href="./memory_snapshot.md">Memory Snapshot</a></td>
    <td>Supports generating device memory snapshots when memory overflow occurs during training.</td>
  </tr>
  <tr>
    <td><a href="./custom_memory_allocator.md">Custom Memory Allocator</a></td>
    <td>Loads a custom NPU memory allocator from the .so file.</td>
  </tr>
  <tr>
    <td><a href="./multistream_memory_reuse.md">Multi-Stream Memory Reuse</a></td>
    <td>In multi-stream scenarios, optimizes memory usage to improve the memory reuse rate.</td>
  </tr>
  <tr>
    <td><a href="./memory_sharing_ipc.md">Memory Sharing (IPC)</a></td>
    <td>Supports cross-process memory sharing to effectively reduce memory consumption.</td>
  </tr>
  <tr>
    <td rowspan="2">Communication Performance Optimization</td>
    <td><a href="./torch_npu_run.md">torch_npu_run</a></td>
    <td>torch_npu_run is an improved version of torchrun for large-scale cluster scenarios, improving cluster link setup performance.</td>
  </tr>
  <tr>
    <td><a href="./ranktable_link_setup.md">Ranktable link setup</a></td>
    <td>Supports establishing a communication domain by configuring the ranktable file.</td>
  </tr>
  <tr>
    <td rowspan="3">Computing Performance Optimization</td>
    <td><a href="./automatic_core_binding.md">Automatic Core Binding</a></td>
    <td>Optimizes TorchNPU dispatch performance by setting coarse-grained or fine-grained core binding.</td>
  </tr>
  <tr>
    <td><a href="./stream_taskqueue_parallel_delivery.md">Stream-Level TaskQueue Parallel Delivery</a></td>
    <td>Each Stream initializes an independent TaskQueue and the corresponding Dequeue thread, implementing a true two-level pipeline parallel delivery mechanism.</td>
  </tr>
  <tr>
    <td><a href="./comp_opt.md">Compilation Optimization</a></td>
    <td>Uses the LTO and PGO compilation optimization technologies of the Bisheng compiler to compile the Python, PyTorch, and TorchNPU components, effectively improving program performance.</td>
  </tr>
  <tr>
    <td rowspan="2">Assisted Error Location</td>
    <td><a href="./feature_value_detection.md">Feature Value Detection</a></td>
    <td>Performs feature value detection for silent data errors based on communication streams to identify precision issues.</td>
  </tr>
  <tr>
    <td><a href="./watchdog.md">WatchDog</a></td>
    <td>Detects errors quickly and stably without affecting the training performance and precision of LLMs.</td>
  </tr>
  <tr>
    <td>Parameter Configuration</td>
    <td><a href="./setting_HCCL_communicator_parameter.md" target="_blank" rel="noopener noreferrer">Configuring HCCL Communication Domain Parameters Through pg_options</a></td>
    <td>Allows different HCCL parameters to be configured for different communication domains.</td>
  </tr>
  <tr>
    <td>torch_npu.npu.NPUGraph</td>
    <td><a href="./pytorch_npugraph_desc.md">torch_npu.npu.NPUGraph</a></td>
    <td>NPUGraph is a static graph capture technique used in Eager Mode (single-operator execution mode). It defines and encapsulates a series of NPU kernel definitions into a unit (that is, an operation graph), and starts multiple NPU operations through a single CPU operation, thereby reducing startup overhead.</td>
  </tr>
  <tr>
    <td>Operator Adaptation</td>
    <td><a href="./custom_operator_adaptation.md">Custom Operator Adaptation Development</a></td>
    <td>Writes and invokes custom operators based on the OpPlugin plugin or C++ extensions.</td>
  </tr>
</tbody></table>
