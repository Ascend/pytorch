# Memory Snapshot

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T07:50:26.007Z pushedAt=2026-06-15T12:00:44.083Z -->

## Introduction

Supports generating a device memory snapshot during training when a memory overflow (OOM) occurs or when the `torch.npu.memory._dump_snapshot` interface is called by the user, and performing visual analysis through an interactive viewer ([memory\_viz](https://pytorch.org/memory_viz)). The snapshot can record the state of allocated NPU memory at any point in time, and can optionally record the history of allocation for that snapshot. This feature is developed based on the [community memory snapshot feature](https://pytorch.org/docs/2.1/torch_cuda_memory.html#understanding-cuda-memory-usage) and supports the usage methods of the community memory snapshot. An illustration of the memory snapshot is shown below:

**Figure 1** Memory usage status diagram  
![figure 1](../figures/memory_usage.png)

The horizontal axis represents the timeline, and the vertical axis represents the current size of occupied device memory. Through the above figure, the memory status in use over time can be intuitively observed. Pan and zoom operations are supported to view smaller memory allocation blocks in the figure. For each allocated memory block, the corresponding stack and allocation information can be viewed.

Additionally, viewing the history of memory allocator states is supported. By selecting each memory allocator event displayed on the left timeline, a visual summary of the memory allocator state at the time of that event operation can be viewed. This summary shows each individual memory segment returned by the program request, as well as how it is divided into individually allocated or free memory blocks based on the actual requested memory size. Similarly, viewing the stack information at the time of memory allocation is also supported. The illustration is shown below:

**Figure 2** Memory allocator status history  
![figure 2](../figures/memory_allocator_status_history.png)

In addition, when a memory snapshot is saved, the current real-time memory occupied by each component during a memory overflow (OOM) (`curMemSize`) and the maximum memory occupied during operation (`memPeakSize`) are both saved to a CSV file in the `OOM_SNAPSHOT_PATH` path. The CSV file can be downloaded and viewed using tools such as Excel.

The recording of memory snapshots is controlled through the environment variables `OOM_SNAPSHOT_ENABLE` and `OOM_SNAPSHOT_PATH`. When used in conjunction with `TASK_QUEUE_ENABLE=2`, the workspace memory usage requested by the taskqueue multi-level pipeline can also be viewed.

## Use Scenario

This feature can be used when it is necessary to analyze NPU memory allocation during model training (for example, when an OOM occurs in the network).

## Usage Guide

- When a memory overflow error occurs in the network, `OOM_SNAPSHOT_ENABLE` can be used to configure whether to save a memory snapshot for analyzing the cause of the memory overflow.
  - When the value is 0, the memory snapshot feature is disabled and memory data is not saved.
  - When the value is 1, upon OOM, current and historical memory usage information is saved, including allocated and freed memory information.
  - When the value is 2, upon OOM, only current memory usage is saved, including allocated and freed memory information.

- When a memory overflow error occurs on the network, the memory data save path can be configured through `OOM_SNAPSHOT_PATH`. This must be used in conjunction with `OOM_SNAPSHOT_ENABLE`.
  - When not configured, the memory data is saved to the current path by default.
  - When configured, the memory data is saved to the specified path.

For details on using this environment variable, refer to the [OOM\_SNAPSHOT\_ENABLE](../environment_variable_reference/OOM_SNAPSHOT_ENABLE.md) section and the [OOM\_SNAPSHOT\_PATH](../environment_variable_reference/OOM_SNAPSHOT_PATH.md) section in the *Environment Variable Reference*.

For usage methods and examples of memory snapshots, refer to the [community documentation](https://pytorch.org/docs/2.7/torch_cuda_memory.html#understanding-cuda-memory-usage). For specific API usage of community memory snapshots, refer to the [API Reference](https://pytorch.org/docs/2.7/torch_cuda_memory.html#snapshot-api-reference).

## Usage Example

- To generate a memory snapshot when a memory overflow (OOM) occurs, configure the following environment variables:

```shell
export OOM_SNAPSHOT_ENABLE=1
export OOM_SNAPSHOT_PATH="/home/usr/"
```

- To save a memory snapshot at any time, call the `torch.npu.memory._dump_snapshot` API:

```python
# enable memory history, which will add tracebacks and event history to snapshots
torch_npu.npu.memory._record_memory_history()

run_your_code()
torch_npu.npu.memory._dump_snapshot("my_snapshot.pickle")
```

## Constraints

- This feature is supported in Ascend Extension for PyTorch 6.0.0 and later versions.
- The feature of saving memory snapshot CSV files upon memory overflow (OOM) is supported only on Ascend HDK 25.5.0 and later versions and CANN commercial 8.5.0 and later versions.
