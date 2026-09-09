# PYTORCH\_NPU\_ALLOC\_CONF

## Function Description

This environment variable controls the behavior of the cache allocator. Configuring this environment variable changes memory usage and may cause performance fluctuations.

The cache allocator uses different memory pools based on the size of the requested memory. Requests smaller than 1 MB use the small block memory pool, and requests of other sizes use the large block memory pool. With the virtual memory feature, the physical memory allocation granularity (`segment_size_mb`) for the large block memory pool defaults to 20 MB, and that for the small block memory pool defaults to 2 MB (non-configurable). In LLM scenarios, the small block memory pool typically has low memory usage. Therefore, certain environment configuration items (`page_size`, `segment_size_mb`) apply only to the large block memory pool.

> [!NOTICE]
>
> Ascend 950DT supports only the following four parameters: `expandable_segments`, `pinned_use_background_threads`, `pin_memory_expandable_segments`, and `pinned_mem_register`. All other parameters are not supported.

Optional parameters:

- `max_split_size_mb:<value>`, the upper limit for splitting memory blocks.

    Memory blocks larger than the configured value are not split during use, which helps reduce memory fragmentation. This option is mainly used for optimization when a model is interrupted due to OOM (Out of Memory) and a large number of inactive split memory blocks exist. The default value of `<value>` is infinite, in MB, meaning that memory blocks of all sizes can be split. The minimum configurable value is greater than 20 MB.

- `garbage_collection_threshold:<value>`, the garbage collection threshold.

    Actively reclaims unused NPU memory blocks. After you set the value threshold (for example, 0.8), if NPU memory capacity usage exceeds the threshold (that is, 80% of the total memory allocated to the NPU application), the cache allocator begins reclaiming NPU memory blocks, first releasing the earliest-allocated and long-unreused memory blocks and avoiding the release of actively reused memory blocks. The value range of `<value>` is (0.0, 1.0). This feature is disabled by default. The garbage collection threshold must be used together with the memory fraction. For the memory fraction, refer to `torch_npu.npu.set_per_process_memory_fraction` in [Custom API](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/en/custom_APIs/overview.md).

- `expandable_segments:<value>`, enables the memory pool segment expansion feature, that is, the virtual memory feature.

    The default value is False. If set to True, this setting instructs the cache allocator to create specific memory block allocations that can be expanded later to better handle scenarios where memory usage size changes frequently. If set to False, the memory pool expandable segment feature is disabled and the original memory allocation method is used.

- `base_addr_aligned_kb:<value>`, the alignment granularity of the memory base address.

    The value range is 0 to 16, and the value must be an integer, in KB. The default value is 16. This parameter takes effect only when `expandable_segments` is set to True. If this parameter is configured as 16, when large block memory (greater than or equal to 2 MB) is requested, the base address of the requested memory is kept aligned to 16 KB as much as possible. If configured as 0, the base address of the requested memory is aligned to 512 Byte.

- `page_size:<value>`, sets the size for requesting large page memory.

    The value supports only 1 GB. Refer to [Configuration Example](#configuration-example). The memory allocation granularity is 1 GB. Amounts less than 1 GB are rounded up to 1 GB, and the allocated amount is always an integer multiple of 1 GB. With the virtual memory feature, this configuration item applies only to the large block memory pool of the cache allocator.

    When this option is not configured, the memory allocation granularity is 2 MB. To request 1 GB of large page memory, 1024/2 = 512 page table entries are occupied. After this option is set, the memory allocation granularity is 1 GB, and 1 GB of large page memory occupies only 1 page table entry. This effectively reduces the number of page table entries and expands the address range covered by the TLB (Translation Lookaside Buffer) cache, improving the performance of scattered access.

    The TLB is a hardware module for high-speed caching in Ascend AI processors, storing the mapping from recently used virtual addresses to physical addresses.

- `segment_size_mb:<value>`, sets the allocation granularity of physical memory under the virtual memory feature.

    The value range is 20 to 512, and the value must be an integer, in MB. When not configured, the default is 20. It takes effect only when `expandable_segments` is set to True, and this configuration item applies only to the large block memory pool of the cache allocator. When configured together with `page_size`, only `page_size` takes effect.
    Increasing `segment_size_mb` can reduce the number of calls to memory allocation and memory mapping interfaces, thereby improving memory allocation efficiency, but it may also bring more memory fragmentation. Therefore, in scenarios where memory usage is at its limit, you are advised to increase this value with caution.

- `roundup_power2_divisions:<value>` or `roundup_power2_divisions:[<size1>:<value1>,<size2>:<value2>,...]`, rounds up the requested allocation size to the nearest power-of-2 segment, thereby reusing memory blocks more efficiently.

    When not configured, the allocation size is aligned up in units of 512 Bytes, which works well for smaller allocation sizes. For larger allocation requests with similar sizes, this strategy may be inefficient. Because each request is allocated to a memory block of a different size, these memory blocks are difficult to reuse, which in turn produces a large number of insufficiently reused memory blocks and wastes memory capacity.

    Two configuration modes are supported:

    - **Single value**: sets the same number of segments for each memory block, for example, "4".
    - **Key-value pair array**: sets the number of segments separately for each power-of-2 interval. For example, when configured as "[256:1,512:2,1024:4,>:8]", 1 segment is set for all allocations below 256 MB, 2 segments for allocations between 256 MB and 512 MB, 4 segments for allocations between 512 MB and 1 GB, and 8 segments for larger allocations.

- `multi_stream_lazy_reclaim:<value>`, delays Events querying during memory allocation in multi-stream scenarios.

    The default value is False, meaning that an Events query is executed for every memory allocation. When set to True, each memory allocation preferentially uses free memory blocks, and an Events query is triggered only when the number of Events exceeds the threshold of 512 or no available memory block is found. By reducing the number of Events queries, CPU resource usage is reduced and host-side performance is improved. This configuration affects only the frequency of Events status queries and does not change the memory release conditions or the memory peak. Memory blocks are released only after all related Events complete.

- `pinned_use_background_threads:<value>`, whether to enable background threads to process events.

    The default value is False, meaning that background threads are not enabled. When set to True, background threads are enabled to query and process events in the background, reducing the blocking time of the main thread.

- `pin_memory_expandable_segments:<value>`, enables the pin_memory memory pool expandable segment feature, that is, the virtual memory feature.

    The default value is False. If set to True, this setting instructs the pin_memory cache allocator to use a physical memory allocation granularity of 20 MB (non-configurable) for the memory pool. The created memory blocks can be expanded later to better handle frequent memory size changes during usage. In addition, statistics related to the pin_memory memory block count are excluded from statistics (default value: 0). If set to False, the pin_memory memory pool expandable segment feature is disabled and the original memory allocation method is used.

- `pinned_mem_register:<value>`, sets whether to enable the host register function for pin_memory memory.

    The default value is False. If set to True, this setting instructs pin_memory to enable the host register function, registering the pin_memory memory mapping as a memory address accessible to the Device. If set to False, the host register function is disabled.

> [!NOTE]
>
> If you use the driver bundled with TorchNPU 6.0.RC3 and later versions (Ascend HDK 24.1.RC3 and later) and enable the virtual memory feature, you can use the single-process multi-device feature. If you use the driver bundled with TorchNPU versions earlier than 6.0.RC3 (Ascend HDK versions earlier than 24.1.RC3) and enable the virtual memory feature, you cannot use the single-process multi-device feature.

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

## Usage Constraints

- The `expandable_segments` feature must be used on Ascend HDK 23.0.0 or later.
- When either `max_split_size_mb` or `garbage_collection_threshold` is set to a non-default value, `expandable_segments` must be set to False.
- The `page_size` feature requires Ascend HDK 25.0.RC1 or later and CANN Commercial Use 8.1.RC1 or later, and supports the following products:
    - Atlas A2 training products
    - Atlas A3 training products

- When the `page_size` feature is configured together with other features, only the `page_size` configuration takes effect. Notes on requesting memory are as follows:
    - When the requested memory is greater than 1 MB:
        - If `page_size` is configured, the memory allocation granularity is 1 GB.
        - If `page_size` is not configured, the memory allocation granularity is 2 MB.

    - When the requested memory is less than or equal to 1 MB: `page_size` does not take effect even if configured, and the memory allocation granularity is 2 MB.
- The `pin_memory_expandable_segments` feature requires TorchNPU 7.3.0 or later, Ascend HDK 25.5.0 or later, and CANN Commercial Use 8.5.0 or later.
- The `pinned_use_background_threads` feature requires TorchNPU 26.0.0 or later and PyTorch 2.8.0 or later.
- Notes on using `pinned_mem_register` are as follows:
    - This feature requires TorchNPU 26.0.0 or later, Ascend HDK 26.0.RC1 or later, and CANN Commercial Use 8.5.0 or later.
    - This feature does not support being configured together with the `pin_memory_expandable_segments` feature.
- Notes on using `multi_stream_lazy_reclaim`:
    - This feature requires TorchNPU 7.3.0 or later.
    - This feature mainly solves system efficiency problems when the host side has a dispatch performance bottleneck in multi-stream scenarios. In single-stream or few-stream scenarios, or when the host is not the performance bottleneck, this feature provides limited benefit.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
