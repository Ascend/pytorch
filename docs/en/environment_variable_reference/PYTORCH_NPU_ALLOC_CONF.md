# PYTORCH_NPU_ALLOC_CONF

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:05:58.464Z pushedAt=2026-06-16T03:14:22.370Z -->

## Feature Description

This environment variable controls the behavior of the cache allocator. Configuring this environment variable will change memory usage and may cause performance fluctuations.

The cache allocator uses different memory pools based on the size of the requested memory. Memory requests smaller than 1 MB use the small block memory pool, while larger requests use the large block memory pool. Under the virtual memory feature, the physical memory granularity (segment_size_mb) for the large block memory pool defaults to 20 MB, and the small block memory pool defaults to 2 MB (non-configurable). In large model scenarios, the small block memory pool typically uses less memory, so some environment configuration items (page_size, segment_size_mb) only affect the large block memory pool.

Optional parameters:

- `max_split_size_mb:<value>`, the upper limit for splitting memory blocks.

    Memory blocks larger than the set value will not be split during use, which helps reduce memory fragmentation. This option is primarily used for optimization when the model is interrupted due to OOM (Out of Memory) and a large number of inactive split memory blocks appear. The default value of <value\> is infinite, in MB, meaning that memory blocks of all sizes can be split, with a minimum setting value greater than 20 MB.

- `garbage_collection_threshold:<value>`, the garbage collection threshold.

    Proactively reclaim unused NPU memory blocks. After setting the value threshold (for example, 0.8), if the NPU memory capacity usage exceeds the threshold (i.e., 80% of the total memory allocated to the NPU app), the cache allocator will start reclaiming NPU memory blocks, prioritizing the release of the earliest allocated and long-unused memory blocks, while avoiding the release of actively reused memory blocks. The value range of <value\> is \(0.0, 1.0\). This feature is not enabled by default. The garbage collection threshold must be used in conjunction with the memory factor. For the memory factor, refer to "torch\_npu.npu.set\_per\_process\_memory\_fraction" in the [Ascend Extension for PyTorch Custom API Reference](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/en/custom_APIs/overview.md).

- `expandable_segments:<value>`, enables the memory pool expandable segments feature, i.e., the virtual memory feature.

Defaults to `False`. If set to `True`, this setting will instruct the cache allocator to create specific memory block allocations that can be subsequently expanded, so as to better handle situations where memory usage size changes frequently. If set to False, the memory pool segment expansion feature is disabled, and the original memory allocation method is used.

- `base_addr_aligned_kb:<value>`, memory base address alignment granularity.

Value Range is 0 to 16. Setting Value Must be an integer. Unit is KB. Default Value is 16. It only takes effect when `expandable_segments` is set to `True`. If this parameter is configured as 16, when applying for large block memory (greater than or equal to 2MB), the system will try to keep the base address of the requested memory 16KB aligned. If configured as 0, the base address of the requested memory is 512B aligned.

- `page_size:<value>`, sets the size for requesting large page memory.

The value only supports 1 GB. Refer to [Configuration Example](#configuration-example). Memory allocation granularity is 1GB. Allocations less than 1 GB are rounded up to 1 GB, always being an integer multiple of 1 GB. Under the virtual memory feature, this configuration item only affects the large block memory pool of the cache allocator.

When this option is not configured, the memory allocation granularity is 2 MB. If you need to allocate 1 GB of large page memory, it will occupy 1024/2 = 512 page tables. After setting this option, the memory allocation granularity is 1 GB, and 1 GB of large page memory occupies only 1 page table, which effectively reduces the number of page tables and expands the address range of the translation lookaside buffer (TLB), thereby improving discrete access performance.

The TLB is a hardware module used for high-speed caching in Ascend AI processors, which stores the mapping of recently used virtual addresses to physical addresses.

- `segment_size_mb:<value>`, sets the physical memory allocation granularity under the virtual memory feature.

Value range: 20 to 512. The setting value must be an integer, in MB. When not configured, it defaults to 20. It takes effect only when `expandable_segments` is set to `True`. This configuration item only affects the large block memory pool of the cache allocator. When configured simultaneously with `page_size`, only `page_size` takes effect. When configured simultaneously with `large_segment_size_mb`, only `large_segment_size_mb` takes effect.
    Increasing `segment_size_mb` can reduce the number of calls to memory allocation and memory mapping interfaces, thereby improving memory allocation efficiency, but it may also cause more memory fragmentation. Therefore, in scenarios with extreme memory usage, exercise caution when increasing this value.

> [!NOTE]
    >
    > `segment_size_mb` is deprecated. It is recommended to use `large_segment_size_mb`. `large_segment_size_mb` supports both virtual memory feature and non-virtual memory feature scenarios, offering more comprehensive functionality.

- `roundup_power2_divisions:<value>` or `roundup_power2_divisions:[<size1>:<value1>,<size2>:<value2>,...]`, rounds up the requested allocation size to the nearest power-of-2 segment, thereby improving memory block reuse efficiency.

    When not configured, the allocation size is aligned upward in units of 512 bytes, which works well for smaller allocation sizes; for larger allocation requests of similar sizes, this strategy may be inefficient. Because each request is allocated to memory blocks of different sizes, making these memory blocks difficult to reuse, which in turn generates a large number of under-reused memory blocks and wastes memory capacity.

    Two configuration methods are supported:

    - **Single value**: sets the same number of segments for each memory block, for example, configured as "4".

    - **Key-value pair array**: sets the number of segments individually for each power-of-2 interval. For example, when configured as "\[256:1,512:2,1024:4,\>:8\]", it means setting 1 segment for all allocations below 256 MB, 2 segments for allocations between 256 MB and 512 MB, 4 segments for allocations between 512 MB and 1 GB, and 8 segments for larger allocations.

- `multi_stream_lazy_reclaim:<value>`, in multi-stream scenarios, lazily queries events during memory allocation.

    Defaults to False, meaning events are queried on every memory allocation. When set to True, each memory allocation preferentially uses free memory blocks, and events are queried only when the number of events exceeds the threshold of 512 or when no available memory block is found. By reducing the number of event queries, CPU resource usage is lowered, thereby improving host-side performance. This configuration only affects the frequency of event status queries, and does not change the conditions for memory release or the peak memory usage. Memory blocks still need to wait for all related events to complete before being released.

- pinned\_use\_background\_threads:<value\>, whether to enable background threads to process events.

    Defaults to False, meaning background threads are not enabled. When set to True, background threads are enabled to perform event querying and processing operations in the background, reducing the blocking time of the main thread.

- `pin_memory_expandable_segments:<value>`, enables the pin_memory memory pool expandable segments feature, that is, the virtual memory feature.

    Defaults to False. If set to True, this setting will instruct the pin_memory cache allocator memory pool to use a physical memory allocation granularity of 20 MB (non-configurable). The created memory blocks can be expanded later, thereby better handling frequent memory size changes during memory usage. Meanwhile, pin_memory memory block count-related statistics will not be included in the statistics (default value: 0). If set to False, the pin_memory memory pool expandable segment feature is disabled, and the original memory allocation method is used.

- `pinned_mem_register:<value>`, sets whether the pin_memory memory enables the host register feature.

    Defaults to False. If set to True, this setting will instruct the pin_memory memory to enable the host register feature, registering the pin_memory memory mapping as a memory address accessible by the Device. If set to False, the host register feature is disabled.

- `large_segment_size_mb:<value>`, sets the segment allocation granularity of the large block memory pool.

    The value must be greater than 10, the setting value must be an integer, the unit is MB, and defaults to 20 when not configured. This configuration item takes effect in both virtual memory feature and non-virtual memory feature scenarios. Under the virtual memory feature, it controls the allocation granularity of physical memory; under the non-virtual memory feature, it controls the segment size for memory allocation in the 1–10 MB range.

    Increasing large_segment_size_mb can reduce the number of calls to memory allocation and memory mapping interfaces, thereby improving memory allocation efficiency, but it may also lead to more memory fragmentation. Therefore, in scenarios with extreme memory usage, exercise caution when increasing this value.

    When configured simultaneously with segment_size_mb, only large_segment_size_mb takes effect. When configured simultaneously with page_size, only page_size takes effect. When configured simultaneously with max_split_size_mb, max_split_size_mb must be greater than or equal to large_segment_size_mb.

- `per_process_memory_fraction:<value>`, limits the proportion of NPU memory that the current process can use.

    The value range is [0.0, 1.0], representing the proportion of available device memory. The default value is 1.0, meaning the process memory usage is not limited. After configuration, the framework calculates the maximum memory available to the current process during initialization based on the total device memory and the configured proportion. Memory allocations exceeding this limit will trigger an OOM (Out of Memory) error.

    This configuration item is suitable for scenarios where multiple processes share the same NPU device. By limiting the proportion of available memory for a process, it prevents a single process from occupying too much memory and causing OOM for other processes.

> [!NOTE]
>
> When users use the driver bundled with Ascend Extension for PyTorch 6.0.RC3 and later versions (Ascend HDK 24.1.RC3 and later versions) and enable the virtual memory feature, they can use the single-process multi-device feature. When users use the driver bundled with Ascend Extension for PyTorch versions earlier than 6.0.RC3 (Ascend HDK versions earlier than 24.1.RC3) and enable the virtual memory feature, they cannot use the single-process multi-device feature.

## Configuration Example<a id="configuration-example"></a>

Example 1:

```bash
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
```

Example 2:

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,base_addr_aligned_kb:16
```

Example 3:

```bash
export PYTORCH_NPU_ALLOC_CONF=page_size:1g
```

Example 4:

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,segment_size_mb:40
```

Example 5:

- Single value example:

    ```bash
    export PYTORCH_NPU_ALLOC_CONF="roundup_power2_divisions:4"
    ```

- Key-value pair array example:

    ```bash
    export PYTORCH_NPU_ALLOC_CONF="roundup_power2_divisions:[256:1,512:2,1024:4,>:8]"
    ```

Example 6:

```bash
export PYTORCH_NPU_ALLOC_CONF=multi_stream_lazy_reclaim:True
```

Example 7:

```bash
export PYTORCH_NPU_ALLOC_CONF=pinned_use_background_threads:True
```

Example 8:

```bash
export PYTORCH_NPU_ALLOC_CONF=pin_memory_expandable_segments:True
```

Example 9:

```bash
export PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True
```

Example 10:

```bash
export PYTORCH_NPU_ALLOC_CONF=large_segment_size_mb:50
```

Example 11:

```bash
export PYTORCH_NPU_ALLOC_CONF=per_process_memory_fraction:0.5
```

## Usage Constraints

- The `expandable_segments` feature requires Ascend HDK 23.0.0 or later.

- When either `max_split_size_mb` or `garbage_collection_threshold` is set to a non-default value, `expandable_segments` must be set to `False`.

- The `page_size` feature requires Ascend HDK 25.0.RC1 and later versions, and CANN commercial 8.1.RC1 and later versions. It supports the following products:

    - <term>Atlas A2 training products</term>

    - <term>Atlas A3 training products</term>

- When the `page_size` feature is configured simultaneously with other features, only the `page_size` configuration takes effect, and the memory allocation notes are as follows:

    - When the requested memory is greater than 1 MB:

        - If page_size is configured, the memory allocation granularity is 1 GB.

        - If page_size is not configured, the memory allocation granularity is 2 MB.

    - When the requested memory is less than or equal to 1 MB: Configuring page_size does not take effect, and the memory allocation granularity is 2 MB.

- The `pin_memory_expandable_segments` feature requires Ascend Extension for PyTorch 7.3.0 and later versions, Ascend HDK 25.5.0 and later versions, and CANN commercial release 8.5.0 and later versions.

- The `pinned_use_background_threads` feature requires Ascend Extension for PyTorch 26.0.0 and later versions and PyTorch 2.8.0 and later versions.

- Usage notes for pinned_mem_register are as follows:
    - Feature requires Ascend Extension for PyTorch 26.0.0 and later versions, Ascend HDK 26.0.rc1 and later versions, and CANN commercial 8.5.0 and later versions.
    - Simultaneous configuration with the pin_memory_expandable_segments feature is not supported.

- Usage notes for `multi_stream_lazy_reclaim`:
    - The feature requires Ascend Extension for PyTorch 7.3.0 and later versions.
    - This feature primarily addresses system efficiency issues when there is a dispatch performance bottleneck on the host side in multi-stream scenarios. In single-stream, few-stream scenarios, or when the host is not the performance bottleneck, this feature provides limited benefits.

- The `large_segment_size_mb` feature requires Ascend Extension for PyTorch 26.1.0 and later versions, and PyTorch 2.11.0 and later versions.

- The `per_process_memory_fraction` feature requires Ascend Extension for PyTorch 26.1.0 and later versions, and PyTorch 2.10.0 and later versions.

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
- <term>Atlas inference series</term>
