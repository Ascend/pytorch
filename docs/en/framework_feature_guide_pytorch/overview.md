# Overview

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T07:50:50.146Z pushedAt=2026-06-15T12:00:44.088Z -->

The Ascend Extension for PyTorch is a deep learning adaptation framework based on Ascend, enabling the Ascend NPU to support the PyTorch framework and providing PyTorch users with the superior computing power of Ascend AI processors.

This plugin inherits the dynamic graph features, native development approach, and architecture of the PyTorch framework to the greatest extent possible, allowing developers to use the Ascend Extension for PyTorch with virtually no changes to their existing development habits and coding styles. Developers can continue using familiar PyTorch interfaces and functions, and seamlessly migrate models to the Ascend platform for training simply by specifying the device as the Ascend NPU, significantly reducing development costs and migration difficulty.

As the deep learning field flourishes, data scales continue to expand and model complexity keeps rising, posing increasing challenges to computing platforms. To address these challenges, the Ascend Extension for PyTorch has carefully crafted a series of unique features in areas such as memory resource optimization, communication performance optimization, computation performance optimization, and assisted error localization. It provides users with an efficient and convenient development tool. For specific features, please refer to [Table 1](#feature-list).

**Table 1**  Feature list <a id="feature-list"></a>
<table style="undefined;table-layout: fixed; width: 1508px"><colgroup>
<col style="width: 293px">
<col style="width: 252px">
<col style="width: 963px">
</colgroup>
<thead>
  <tr>
    <th>Feature Category<br></th>
    <th>Feature Name<br></th>
    <th>Feature Description<br></th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5">Memory resource optimization</td>
    <td><a href="./virtual_memory.md">Virtual Memory</a></td>
    <td>Dynamically adjusts memory block sizes to reduce memory fragmentation.</td>
  </tr>
  <tr>
    <td><a href="./memory_snapshot.md">Memory Snapshot</a></td>
    <td>Supports generating device memory snapshots when out-of-memory occurs during training.</td>
  </tr>
  <tr>
    <td><a href="./custom_memory_allocator.md">Custom Memory Allocator</a></td>
    <td>Loads a custom NPU memory allocator from a .so file.</td>
  </tr>
  <tr>
    <td><a href="./multistream_memory_reuse.md">Multi-Stream Memory Reuse</a></td>
    <td>Optimizes memory usage in multi-stream scenarios to improve memory reuse rates.</td>
  </tr>
  <tr>
    <td><a href="./memory_sharing_ipc.md">Memory Sharing (IPC)</a></td>
    <td>Supports cross-process memory sharing to effectively reduce memory consumption.</td>
  </tr>
  <tr>
    <td rowspan="2">Communication performance optimization</td>
    <td><a href="./torch_npu_run.md">torch_npu_run</a></td>
    <td>An improved version of torchrun for large-scale cluster scenarios, enhancing cluster link establishment performance.</td>
  </tr>
  <tr>
    <td><a href="./ranktable_link_setup.md">Ranktable Link Setup</a></td>
    <td>Supports establishing communication domains via ranktable file configuration.</td>
  </tr>
  <tr>
    <td rowspan="2">Computation performance optimization</td>
    <td><a href="./automatic_core_binding.md">Automatic Core Binding</a></td>
    <td>Optimizes the dispatch performance of Ascend Extension for PyTorch by setting coarse/fine-grained core binding.</td>
  </tr>
  <tr>
    <td><a href="./stream_taskqueue_parallel_delivery.md">Stream-Level TaskQueue Parallel Dispatch</a></td>
    <td>Each stream initializes an independent TaskQueue and corresponding dequeue thread, implementing a true two-level pipeline parallel dispatch mechanism.</td>
  </tr>
  <tr>
    <td rowspan="2">Auxiliary error localization</td>
    <td><a href="./feature_value_detection.md">Feature Value Detection</a></td>
    <td>Performs feature value detection for silent data errors based on communication streams to identify precision issues.</td>
  </tr>
  <tr>
    <td><a href="./watchdog.md">WatchDog</a></td>
    <td>Quickly and stably detects errors without affecting the training performance and precision of large models.</td>
  </tr>
  <tr>
    <td>Parameter configuration</td>
    <td><a href="./setting_HCCL_communicator_parameter.md" target="_blank" rel="noopener noreferrer">Configuring HCCL Communication Domain Parameters via pg_options</a></td>
    <td>Allows different HCCL configurations to be set for different communication domains.</td>
  </tr>
  <tr>
    <td>Operator adaptation</td>
    <td><a href="./custom_operator_adaptation.md">Custom Operator Adaptation Development</a></td>
    <td>Writes and invokes custom operators based on the OpPlugin or C++ extensions approach.</td>
  </tr>
</tbody></table>
