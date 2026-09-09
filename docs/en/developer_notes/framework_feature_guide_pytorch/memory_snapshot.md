# Memory Snapshot

## Introduction

The memory snapshot feature supports generating device memory snapshots when a Out Of Memory(OOM) occurs during training or when the user calls the `torch.npu.memory._dump_snapshot` interface, and enables visual analysis through an interactive viewer ([memory_viz](https://pytorch.org/memory_viz)). The snapshot can record the state of allocated NPU memory at any point in time, and can optionally record the history of memory allocation operations. This feature is developed based on the community [memory snapshot feature](https://pytorch.org/docs/2.1/torch_cuda_memory.html#understanding-cuda-memory-usage) and supports the usage patterns of the community memory snapshot. An illustration of the memory snapshot is shown below:

**Figure 1**  Schematic diagram of memory usage status  
![figure1](../figures/memory_usage.png)

The horizontal axis represents the timeline, and the vertical axis represents the current device memory usage. From the figure, you can intuitively observe the memory usage status over time. You can pan and zoom to inspect smaller memory allocation blocks in the figure. For each allocated memory block, you can view the corresponding stack trace and allocation information.

You can also view the history of memory allocator states. By selecting each memory allocator event displayed on the left timeline, you can view a visual summary of the memory allocator state at the time the event was executed. This summary shows each individual memory segment returned by the allocation request, as well as how the segments are divided into individually allocated or free memory blocks based on the actual requested memory size. Similarly, you can view the stack information at the time of memory allocation. The result is shown in the following figure:

**Figure 2**  Schematic diagram of memory allocator state history  
![figure2](../figures/memory_allocator_status_history.png)

In addition, when the memory snapshot is saved, the memory currently occupied in real time by each component at the time of the Out Of Memory(OOM) (`curMemSize`) and the maximum memory occupied during execution (`memPeakSize`) are both saved to a CSV file in the `OOM_SNAPSHOT_PATH` path. You can download the CSV file and view it using tools such as Excel.

The environment variables `OOM_SNAPSHOT_ENABLE` and `OOM_SNAPSHOT_PATH` are used to control the recording of memory snapshots. When used together with `TASK_QUEUE_ENABLE=2`, you can also view the workspace memory usage of the multi-level taskqueue pipeline.

## Use Scenarios

During model training, if you need to analyze NPU memory allocation (for example, when an OOM error occurs in the network), you can use this feature.

## Usage Guide

- When an out-of-memory error occurs in the network, you can configure whether to save a memory snapshot through `OOM_SNAPSHOT_ENABLE` to analyze the cause of the out-of-memory error.
    - When set to 0, the memory snapshot feature is disabled and no memory data is saved.
    - When set to 1, upon OOM, the current and historical memory usage information is saved, including allocated and freed memory information.
    - When set to 2, only the current memory usage is saved upon OOM, including allocated and freed memory information.

- When an out-of-memory error occurs in the network, you can configure the memory data save path through `OOM_SNAPSHOT_PATH`. This must be used together with `OOM_SNAPSHOT_ENABLE`.
    - When not configured, memory data is saved to the current path by default.
    - When configured, memory data is saved to the specified path.

For details on using this environment variable, refer to the "[OOM_SNAPSHOT_ENABLE](../environment_variable_reference/OOM_SNAPSHOT_ENABLE.md)" section in *Environment Variable Reference* and the "[OOM_SNAPSHOT_PATH](../environment_variable_reference/OOM_SNAPSHOT_PATH.md)" section in *Environment Variable Reference*.

For usage methods and examples of memory snapshots, refer to the [community documentation](https://pytorch.org/docs/2.7/torch_cuda_memory.html#understanding-cuda-memory-usage). For specific usage of the community memory snapshot APIs, see the [API reference](https://pytorch.org/docs/2.7/torch_cuda_memory.html#snapshot-api-reference).

## Usage Example

- To generate a memory snapshot when a Out Of Memory(OOM) occurs, configure the following environment variables:

```shell
export OOM_SNAPSHOT_ENABLE=1
export OOM_SNAPSHOT_PATH="/home/usr/"
```

- To save a memory snapshot at any time, call the `torch.npu.memory._dump_snapshot` API:

```python
# Enable memory history, which will add tracebacks and event history to snapshots
torch_npu.npu.memory._record_memory_history()

run_your_code()
torch_npu.npu.memory._dump_snapshot("my_snapshot.pickle")
```

## Constraints

- TorchNPU 6.0.0 and later versions support this feature.
- The feature of saving memory snapshot CSV files upon Out Of Memory(OOM) is supported only on Ascend HDK 25.5.0 and later versions, as well as CANN commercial 8.5.0 and later versions.
