# PYTORCH_NPU_ALLOC_CONF

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-08T10:23:31.276Z pushedAt=2026-07-08T10:47:16.879Z -->

## Feature Description

This environment variable controls the behavior of the cache allocator. Configuring this environment variable will change memory usage and may cause performance fluctuations.

The cache allocator uses different memory pools based on the size of the requested memory. For requests smaller than 1 MB, the small block memory pool is used; otherwise, the large block memory pool is used. With the virtual memory feature enabled, the physical memory granularity (segment_size_mb) for the large block memory pool defaults to 20 MB, while the small block memory pool defaults to 2 MB (non-configurable). In large model scenarios, the small block memory pool typically has low memory usage, so certain environment configuration items (page_size, segment_size_mb) only apply to the large block memory pool.

Parameters:

- `max_split_size_mb:<value>`, the upper limit for splitting memory blocks.

    Memory blocks larger than the specified value will not be split during usage, which helps reduce memory fragmentation. This option is primarily used for optimization when a model is interrupted due to OOM (Out of Memory) and a large number of inactive split memory blocks are present. The default value of `<value>` is infinite, in MB, meaning that memory blocks of all sizes can be split. The minimum configurable value is greater than 20 MB.

- `garbage_collection_threshold:<value>`, the garbage collection threshold.

    Actively reclaims unused NPU memory blocks. After setting the value threshold (for example, 0.8), if the NPU memory capacity usage exceeds the threshold (i.e., 80% of the total memory allocated to the NPU app), the cache allocator will begin reclaiming NPU memory blocks, prioritizing the release of the earliest-allocated and long-unreused memory blocks while avoiding the release of actively reused memory blocks. The value range of `<value>` is (0.0, 1.0). This feature is disabled by default. The garbage collection threshold must be used in conjunction with the memory fraction. For the memory fraction, refer to "torch_npu.npu.set_per_process_memory_fraction" in *[Ascend Extension for PyTorch Custom API Reference](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/en/custom_APIs/overview.md)*.

- `expandable_segments:<value>`, enables the memory pool segment expansion feature, that is, the virtual memory feature.

The default value is False. If set to True, this setting instructs the cache allocator to create specific memory block allocations that can be subsequently expanded, enabling better handling of scenarios where memory usage size changes frequently. If set to False, the memory pool expandable segment feature is disabled, and the original memory allocation method is used.

- `base_addr_aligned_kb:<value>`, the alignment granularity of the memory base address.

The value range is 0 to 16, and the value must be an integer, in KB. The default value is 16. This parameter takes effect only when expandable_segments is set to True. If this parameter is configured as 16, when applying for large block memory (greater than or equal to 2 MB), the allocator will attempt to keep the base address of the allocated memory aligned to 16 KB. If configured as 0, the base address of the allocated memory is aligned to 512 B.

- `page_size:<value>`, sets the size for applying for large page memory.

The value supports only 1 GB. Refer to [Configuration Example](#configuration-example). The memory allocation granularity is 1 GB, and any amount less than 1 GB is rounded up to 1 GB, always being an integer multiple of 1 GB. Under the virtual memory feature, this configuration item applies only to the large block memory pool of the cache allocator.

    When this option is not configured, the memory allocation granularity is 2 MB. If a 1 GB large page memory needs to be allocated, it will occupy 1024/2 = 512 page table entries. After setting this option, the memory allocation granularity becomes 1 GB, and a 1 GB large page memory occupies only 1 page table entry, effectively reducing the number of page table entries and expanding the address range covered by the TLB (Translation Lookaside Buffer) cache, thereby improving the performance of scattered access.

    The TLB is a hardware module used for high-speed caching in Ascend AI processors, which stores the mapping of recently used virtual addresses to physical addresses.

- `segment_size_mb:<value>`, sets the allocation granularity of physical memory under the virtual memory feature.

    The value range is 20 to 512, and the value must be an integer in MB. When not configured, the default is 20. It takes effect only when expandable_segments is set to True, and this configuration item applies only to the large block memory pool of the cache allocator. When configured together with page_size, only page_size takes effect.
    Increasing segment_size_mb can reduce the number of calls to memory allocation and memory mapping interfaces, thereby improving memory allocation efficiency, but it may also introduce more memory fragmentation. Therefore, in scenarios where memory usage is at its limit, exercise caution when increasing this value.

- `roundup_power2_divisions:<value>` or `roundup_power2_divisions:[<size1>:<value1>,<size2>:<value2>,...]`, rounds up the requested allocation size to the nearest power-of-two segment, thereby enabling more efficient reuse of memory blocks.

When not configured, the allocation size is aligned upward in units of 512 bytes, which works well for smaller allocation sizes; for larger allocation requests of similar sizes, this strategy may be inefficient. Each request is assigned to memory blocks of different sizes, making these blocks difficult to reuse, which in turn generates a large number of underutilized memory blocks and wastes memory capacity.

Two configuration modes are supported:

- **Single value**: sets the same number of segments for each memory block, for example, configured as "4".
- **Key-value pair array**: sets the number of segments individually for each power-of-2 interval. For example, when configured as "[256:1,512:2,1024:4,>:8]", it means setting 1 segment for all allocations below 256 MB, 2 segments for allocations between 256 MB and 512 MB, 4 segments for allocations between 512 MB and 1 GB, and 8 segments for larger allocations.

- `multi_stream_lazy_reclaim:<value>`, delays Events querying during memory allocation in multi-stream scenarios.

The default value is False, meaning that an Events query is executed for every memory allocation. When set to True, each memory allocation preferentially uses free memory blocks, and an Events query is triggered only when the number of Events exceeds the threshold of 512 or when no available memory block is found. By reducing the frequency of Events queries, CPU resource usage is lowered, improving host-side performance. This configuration only affects the frequency of Events status queries; it does not change the conditions for memory release or the peak memory usage. Memory blocks must still wait for all related Events to complete before being released.

- `pinned_use_background_threads:<value>`, whether to enable background threads for processing events.

The default value is False, meaning background threads are not enabled. When set to True, background threads are enabled to execute event querying and processing operations in the background, reducing the blocking time of the main thread.

- `pin_memory_expandable_segments:<value>`, enables the pin_memory memory pool expandable segment feature, that is, the virtual memory feature.

The default value is False. If set to True, this setting instructs the pin_memory cache allocator to use a physical memory allocation granularity of 20 MB (non-configurable) for the memory pool, and the created memory blocks can be subsequently expanded to better handle frequent memory size changes during usage. Meanwhile, pin_memory memory block count-related statistics are excluded from statistical reporting (default value: 0). If set to False, the pin_memory memory pool expandable segment feature is disabled, and the original memory allocation method is used.

- `pinned_mem_register:<value>`, sets whether to enable the host register function for pin_memory.

    The default value is False. If set to True, this setting instructs pin_memory to enable the host register function, registering the pin_memory mapping as a memory address accessible to the Device. If set to False, the host register function is disabled.

> [!NOTE]  
>
> When using drivers compatible with Ascend Extension for PyTorch 6.0.RC3 or later (Ascend HDK 24.1.RC3 or later) with the virtual memory feature enabled, the single-process multi-device feature is available. When using drivers compatible with Ascend Extension for PyTorch versions earlier than 6.0.RC3 (Ascend HDK versions earlier than 24.1.RC3) with the virtual memory feature enabled, the single-process multi-device feature is not available.

## Configuration Example

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

- Array of key-value pairs example:

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

## Usage Constraints

- The expandable_segments feature must be used on Ascend HDK 23.0.0 or later.
- When either max_split_size_mb or garbage_collection_threshold is set to a non-default value, expandable_segments must be set to False.
- The page_size feature requires Ascend HDK 25.0.RC1 or later and CANN Commercial Use 8.1.RC1 or later, and supports the following products:
  - Atlas A2 training products
  - Atlas A3 training products

- The page_size feature cannot be configured simultaneously with other features, and the following considerations apply to memory allocation:
  - When the requested memory is greater than 1 MB:
    - If page_size is configured, the memory allocation granularity is 1 GB.
    - If page_size is not configured, the memory allocation granularity is 2 MB.

  - When the requested memory is less than or equal to 1 MB: configuring page_size does not take effect, and the memory allocation granularity is 2 MB.
- The pin_memory_expandable_segments feature requires Ascend Extension for PyTorch 7.3.0 or later, Ascend HDK 25.5.0 or later, and CANN 8.5.0 or later (Commercial Use).
- The pinned_use_background_threads feature requires Ascend Extension for PyTorch 26.0.0 or later and PyTorch 2.8.0 or later.
- Notes on using pinned_mem_register are as follows:
  - This feature requires Ascend Extension for PyTorch 26.0.0 or later, Ascend HDK 26.0.rc1 or later, and CANN 8.5.0 or later (Commercial Use).
  - It is not supported to be configured simultaneously with the pin_memory_expandable_segments feature.
- Notes on using multi_stream_lazy_reclaim:
  - Feature requirement: Ascend Extension for PyTorch 7.3.0 or later is required.
  - This feature primarily addresses system efficiency issues when there is a dispatch performance bottleneck on the host side in multi-stream scenarios. In single-stream or few-stream scenarios, or when the host is not the performance bottleneck, this feature provides limited benefit.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
  