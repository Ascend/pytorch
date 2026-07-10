# Overview

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-08T10:21:40.153Z pushedAt=2026-07-08T10:47:16.860Z -->

Ascend Extension for PyTorch is a deep learning adaptation framework based on Ascend, enabling Ascend NPU to support the PyTorch framework and providing PyTorch users with the exceptional computing power of Ascend AI processors.

This plugin inherits the dynamic graph features, native development paradigm, and architecture of the PyTorch framework to the greatest extent possible, allowing developers to use Ascend Extension for PyTorch with virtually no changes to their existing development habits or coding style. Developers can continue using familiar PyTorch APIs and functions, and seamlessly migrate models to the Ascend platform for training simply by specifying Ascend NPU as the device, significantly reducing development costs and migration difficulty.

As deep learning continues to flourish, data scales keep expanding, and model complexity steadily increases, the challenges posed to computing platforms are growing ever more demanding. In response to these challenges, Ascend Extension for PyTorch has meticulously crafted a series of unique features spanning Memory Resource Optimization, Communication Performance Optimization, Computational Performance Optimization, and Assisted Error Location. These features provide users with an efficient and convenient development tool. For details on specific features, see Table 1.

**Table 1**  Feature list

|Feature Category|Feature Name|Feature Description|
|--|--|--|
|Memory Resource Optimization|[Virtual Memory](virtual_memory.md)|Dynamically adjusts memory block sizes to reduce memory fragmentation.|
|Memory Resource Optimization|[Memory Snapshot](memory_snapshot.md)|Supports generating device memory snapshots upon memory overflow during training.|
|Memory Resource Optimization|[Custom Memory Allocator](custom_memory_allocator.md)|Loads a custom NPU memory allocator from a .so file.|
|Memory Resource Optimization|[Multi-Stream Memory Reuse](multistream_memory_reuse.md)|Optimizes memory usage in multi-stream scenarios to improve memory reuse rates.|
|Memory Resource Optimization|[Memory Sharing (IPC)](memory_sharing_ipc.md)|Supports cross-process memory sharing to effectively reduce memory consumption.|
|Communication Performance Optimization|[torch_npu_run](torch_npu_run.md)|An improved version of torchrun for large-scale cluster scenarios, enhancing cluster link setup performance.|
|Communication Performance Optimization|[Ranktable Link Setup](ranktable_link_setup.md)|Supports establishing communication domains via ranktable file configuration.|
|Computational Performance Optimization|[Automatic Core Binding](automatic_core_binding.md)|Optimizes Ascend Extension for PyTorch dispatch performance by configuring coarse-grained or fine-grained core binding.|
|Computational Performance Optimization|[Stream-Level TaskQueue Parallel Dispatch](stream_taskqueue_parallel_delivery.md)|Each stream initializes an independent TaskQueue and corresponding dequeue thread, implementing a true two-level pipeline parallel dispatch mechanism.|
|Assisted Error Location|[Feature Value Detection](feature_value_detection.md)|Performs feature value detection for silent data errors based on communication streams to identify precision issues.|
|Assisted Error Location|[WatchDog](watchdog.md)|Enables fast and stable error detection without compromising the training performance and precision of large models.|
|Parameter Configuration|[Configuring HCCL Communication Domain Parameters via pg_options](setting_HCCL_communicator_parameter.md)|Allows different HCCL configurations to be set for different communication domains.|
|PyTorch Graph Mode|[PyTorch Compilation Mode (torch.compile)](pytorch_compilation_mode.md)|Ascend Extension for PyTorch supports torch.compile() from version 2.6.0 onward, significantly accelerating model training and inference tasks through "dynamic graph capture + static graph optimization + efficient code generation."|
|PyTorch Graph Mode|[NPUGraph](pytorch_npugraph_desc.md)|NPUGraph is a static graph capture technique that converts dynamic PyTorch operations into a fixed computation graph, improving NPU execution efficiency.|
|Operator Adaptation|[Custom Operator Adaptation Development](custom_operator_adaptation.md)|Develops and invokes custom operators based on OpPlugin or C++ extensions.|
