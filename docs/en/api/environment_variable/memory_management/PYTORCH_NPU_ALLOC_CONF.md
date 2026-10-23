# PYTORCH\_NPU\_ALLOC\_CONF

## Function Description

This environment variable controls the behavior of the cache allocator. Configuring this environment variable changes memory usage and may cause performance fluctuations.

> [!CAUTION]  
>
> Starting from TorchNPU 26.2.0 and PyTorch 2.10.0 and later versions, torch\_npu also supports configuring caching allocator parameters through the [PYTORCH_ALLOC_CONF](PYTORCH_ALLOC_CONF.md) environment variable. The configuration method and supported parameters are the same as PYTORCH\_NPU\_ALLOC\_CONF. Configure either one. Configuring both at the same time reports an error and exits the program. It is recommended to use the PYTORCH\_ALLOC\_CONF environment variable to configure caching allocator behavior.

The cache allocator uses different memory pools based on the size of the requested memory. Requests smaller than 1 MB use the small block memory pool, and requests of other sizes use the large block memory pool. With the virtual memory feature, the physical memory allocation granularity (`segment_size_mb`) for the large block memory pool defaults to 20 MB, and that for the small block memory pool defaults to 2 MB (non-configurable). In LLM scenarios, the small block memory pool typically has low memory usage. Therefore, certain environment configuration items (`page_size`, `segment_size_mb`) apply only to the large block memory pool.

Optional parameters:

- `max_split_size_mb:<value\>`, the upper limit for splitting memory blocks.

    Memory blocks larger than the configured value are not split during use, which helps reduce memory fragmentation. This option is mainly used for optimization when a model is interrupted due to OOM (Out of Memory) and a large number of inactive split memory blocks exist. The default value of `<value\>` is infinite, in MB, meaning that memory blocks of all sizes can be split. The minimum configurable value is greater than 20 MB.

- `garbage_collection_threshold:<value\>`, the garbage collection threshold.

    Actively reclaims unused NPU memory blocks. After the value threshold is set (for example, 0.8), if NPU memory capacity usage exceeds the threshold (that is, 80% of the total memory allocated to the NPU application), the caching allocator starts reclaiming NPU memory blocks, preferentially releasing the blocks that were allocated first and have not been reused for a long time, while avoiding releasing blocks that are actively reused. The <value\> value range is \(0.0,1.0\). This feature is disabled by default. The garbage collection threshold must be used together with the memory factor. For the memory factor, see "torch\_npu.npu.set\_per\_process\_memory\_fraction" in [Custom APIs](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/en/custom_APIs/overview.md).

- `expandable_segments:<value\>`, enables the memory pool segment expansion feature, that is, the virtual memory feature.

    The default value is False. If set to True, this setting instructs the cache allocator to create specific memory block allocations that can be expanded later to better handle scenarios where memory usage size changes frequently. If set to False, the memory pool expandable segment feature is disabled and the original memory allocation method is used.

- `base_addr_aligned_kb:<value\>`, the alignment granularity of the memory base address.

    The value range is 0 to 16, and the value must be an integer, in KB. The default value is 16. This parameter takes effect only when `expandable_segments` is set to True. If this parameter is configured as 16, when large block memory (greater than or equal to 2 MB) is requested, the base address of the requested memory is kept aligned to 16 KB as much as possible. If configured as 0, the base address of the requested memory is aligned to 512 B.

<!-- npu="950,A3,910b" id7 -->
- `page_size:<value\>`, sets the size for requesting large page memory.

    The value supports only 1 GB. Refer to [Configuration Example](#configuration-example). The memory allocation granularity is 1 GB. Amounts less than 1 GB are rounded up to 1 GB, and the allocated amount is always an integer multiple of 1 GB. With the virtual memory feature, this configuration item applies only to the large block memory pool of the cache allocator.

    When this option is not configured, the memory allocation granularity is 2 MB. To request 1 GB of large page memory, 1024/2 = 512 page table entries are occupied. After this option is set, the memory allocation granularity is 1 GB, and 1 GB of large page memory occupies only 1 page table entry. This effectively reduces the number of page table entries and expands the address range covered by the TLB (Translation Lookaside Buffer) cache, improving the performance of scattered access.

    The TLB is a hardware module for high-speed caching in Ascend AI processors, storing the mapping from recently used virtual addresses to physical addresses.
<!-- end id7 -->

- `segment_size_mb:<value\>`, sets the allocation granularity of physical memory under the virtual memory feature.

    The value range is 20\~512. The configured value must be an integer, in MB, and defaults to 20 when not configured. It takes effect only when expandable\_segments is set to True. This configuration item applies only to the large-block memory pool of the caching allocator. When configured together with page\_size, only page\_size takes effect. When configured together with large\_segment\_size\_mb, only large\_segment\_size\_mb takes effect.
    Increasing `segment_size_mb` can reduce the number of calls to memory allocation and memory mapping interfaces, thereby improving memory allocation efficiency, but it may also bring more memory fragmentation. Therefore, in scenarios where memory usage is at its limit, you are advised to increase this value with caution.

    > [!NOTE]
    >
    > segment\_size\_mb is deprecated; large\_segment\_size\_mb is recommended. large\_segment\_size\_mb supports both virtual memory and non-virtual memory scenarios, providing more comprehensive functionality.

- `roundup_power2_divisions:<value\>` or `roundup_power2_divisions:[<size1\>:<value1\>,<size2\>:<value2\>,...]`, rounds up the requested allocation size to the nearest power-of-2 segment, thereby reusing memory blocks more efficiently.

    When not configured, the allocation size is aligned up in units of 512 bytes, which works well for smaller allocation sizes. For larger allocation requests with similar sizes, this strategy may be inefficient. Because each request is allocated to a memory block of a different size, these memory blocks are difficult to reuse, which in turn produces a large number of insufficiently reused memory blocks and wastes memory capacity.

    Two configuration modes are supported:

    - **Single value**: sets the same number of segments for each memory block, for example, "4".
    - **Key-value pair array**: sets the number of segments separately for each power-of-2 interval. For example, when configured as "[256:1,512:2,1024:4,>:8]", 1 segment is set for all allocations below 256 MB, 2 segments for allocations between 256 MB and 512 MB, 4 segments for allocations between 512 MB and 1 GB, and 8 segments for larger allocations.

- `multi_stream_lazy_reclaim:<value\>`, delays Events querying during memory allocation in multi-stream scenarios.

    The default value is False, meaning that an Events query is executed for every memory allocation. When set to True, each memory allocation preferentially uses free memory blocks, and an Events query is triggered only when the number of Events exceeds the threshold of 512 or no available memory block is found. By reducing the number of Events queries, CPU resource usage is reduced and host-side performance is improved. This configuration affects only the frequency of Events status queries and does not change the memory release conditions or the memory peak. Memory blocks are released only after all related Events complete.

- `pinned_use_background_threads:<value\>`, whether to enable background threads to process events.

    The default value is False, meaning that background threads are not enabled. When set to True, background threads are enabled to query and process events in the background, reducing the blocking time of the main thread.

- `pin_memory_expandable_segments:<value\>`, enables the pin_memory memory pool expandable segment feature, that is, the virtual memory feature.

    The default value is False. If set to True, this setting instructs the pin_memory cache allocator to use a physical memory allocation granularity of 20 MB (non-configurable) for the memory pool. The created memory blocks can be expanded later to better handle frequent memory size changes during usage. In addition, statistics related to the pin_memory memory block count are excluded from statistics (default value: 0). If set to False, the pin_memory memory pool expandable segment feature is disabled and the original memory allocation method is used.

- `pinned_mem_register:<value\>`, sets whether to enable the host register function for pin_memory memory.

    The default value is False. If set to True, this setting instructs pin_memory to enable the host register function, registering the pin_memory memory mapping as a memory address accessible to the Device. If set to False, the host register function is disabled.

- large\_segment\_size\_mb:<value\>: Sets the segment allocation granularity of the large-block memory pool.

    The value must be greater than 10. The configured value must be an integer, in MB, and defaults to 20 when not configured. This configuration item applies to both virtual memory and non-virtual memory scenarios. Under the virtual memory feature, it controls the granularity of physical memory requests. Under the non-virtual memory feature, it controls the segment size for memory requests in the 1~10MB range.

    Increasing large\_segment\_size\_mb can reduce the number of calls to the memory request and memory mapping interfaces, thereby improving memory request efficiency, but it may also cause more memory fragmentation. Therefore, in scenarios where memory usage is at its limit, increase this value with caution.

    When configured together with segment\_size\_mb, only large\_segment\_size\_mb takes effect. When configured together with page\_size, only page\_size takes effect. When configured together with max\_split\_size\_mb, max\_split\_size\_mb >= large\_segment\_size\_mb is required.

- per\_process\_memory\_fraction:<value\>: Limits the proportion of NPU device memory available to the current process.

    The value range is \[0.0,1.0\], indicating the proportion of available device memory. The default value is 1.0, which means process device memory usage is not limited. After configuration, during initialization, the framework calculates the maximum device memory available to the current process based on the total device memory and the configured proportion. Memory requests exceeding the limit trigger an OOM (Out of Memory) error.

    This configuration item applies to scenarios where multiple processes share the same NPU device. By limiting the proportion of available device memory of each process, it prevents a single process from occupying too much device memory and causing OOM in other processes.

- throw\_on\_npumalloc\_oom:<value\>: Rejects memory requests preventively before an OOM occurs, skipping the memory allocation retry process.

    The value is True or False, and defaults to False. When the default is used or the value is set to False, if memory allocation through the underlying ACL interface fails, some free memory is released, necessary Host-Device synchronization is performed, and memory allocation is retried. This retry process has a certain impact on performance. When set to True, before calling the underlying ACL interface to allocate memory, the caching allocator first checks whether "the memory size already allocated by the current process + the current request size" exceeds the device memory limit. If it does, the current request is directly rejected and an OutOfMemoryError exception is raised, without entering the memory allocation retry process.

    throw\_on\_npumalloc\_oom does not take effect in the following two cases:
    - per\_process\_memory\_fraction is configured to be less than 1.0 using the environment variables PYTORCH\_NPU\_ALLOC\_CONF or PYTORCH\_ALLOC\_CONF.
    - fraction is set to less than 1.0 by calling the torch\_npu.npu.set\_per\_process\_memory\_fraction\(\) interface.

    When throw\_on\_npumalloc\_oom is set to True and a memory request is rejected, the number of rejected memory requests is accumulated in the torch\_npu.npu.memory\_stats\(\)\[\'num\_oom\_rejections\'\] field, which can be cleared by calling the torch\_npu.npu.reset\_accumulated\_memory\_stats\(\) interface. In this case, the environment variable OOM\_SNAPSHOT\_ENABLE does not take effect.

- pinned\_max\_round\_threshold\_mb:<value\>: The threshold for determining whether the pinned memory allocation size is rounded up to a power of 2.

    The value is a positive integer, in MB. It is disabled by default, meaning power-of-2 rounding up is performed for all pinned memory allocations. After configuration, if the allocation size is less than or equal to this threshold, rounding up is still performed. If the allocation size is greater than this threshold, rounding is skipped and the allocation is made at the exact requested size, thereby reducing memory waste caused by rounding of large pinned memory blocks. For example, when the threshold is set to 128MB, a 129MB allocation request uses 129MB instead of being rounded up to 256MB.

    Rounding up is intended to make memory blocks easier to reuse later. Blocks that do not enter the cache do not need alignment. Therefore, regardless of how this option is set, allocations exceeding pinned\_max\_cached\_size\_mb are never rounded up.

- pinned\_max\_cached\_size\_mb:<value\>: The threshold for determining whether released pinned memory is cached to the free list for reuse.

    The value is a positive integer, in MB. It is disabled by default, meaning all released pinned memory blocks enter the free list cache for reuse. After configuration, if the block size is greater than this threshold, the block is immediately returned to the OS upon release and no longer enters the free list cache (and is no longer rounded up), thereby reducing peak memory usage in scenarios where large pinned memory blocks are used infrequently.

- max\_non\_split\_rounding\_mb:<value\>: A device memory reuse control item that works with max\_split\_size\_mb to control the device memory block size and reuse.

    The value is a positive integer, in MB, with a default of 20M. After a device memory block is freed, it is not necessarily physically released and is assumed to be retained for the next allocation. Assume the retained device memory size of this block is A, max\_non\_split\_rounding\_mb is B, and max\_split\_size\_mb is C, and a new memory allocation D is required.

    When D is greater than C, calculate A-D:
    - If A-D is less than or equal to B, no new allocation is required; reuse A.
    - If A-D is greater than B, new device memory needs to be allocated.

    When D is less than or equal to C, this variable does not take effect.

- release\_lock\_on\_npumalloc:<value\>: A device memory allocation lock control item that decides whether to release some locks when allocating device memory. The controlled items remain consistent with the native community.

    The value is of bool type: True or False, and defaults to False. After configuration, device memory allocation releases some locks, which can alleviate host computing overhead pressure but may cause more resource occupation. Custom-developed device memory management may have compatibility issues; use it with caution.

> [!NOTE]  
>
> If you use the driver bundled with TorchNPU 6.0.RC3 and later versions (Ascend HDK 24.1.RC3 and later) and enable the virtual memory feature, you can use the single-process multi-device feature. If you use the driver bundled with TorchNPU versions earlier than 6.0.RC3 (Ascend HDK versions earlier than 24.1.RC3) and enable the virtual memory feature, you cannot use the single-process multi-device feature.

## Configuration Example <a id="configuration-example"></a> 

Example 1:

```bash
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
```

Example 2:

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,base_addr_aligned_kb:16
```

<!-- npu="950,A3,910b" id8 -->
Example 3:

```bash
export PYTORCH_NPU_ALLOC_CONF=page_size:1g
```
<!-- end id8 -->

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

Example 12:

```bash
export PYTORCH_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
```

Example 13:

```bash
export PYTORCH_NPU_ALLOC_CONF=throw_on_npumalloc_oom:True
```

Example 14:

```bash
export PYTORCH_NPU_ALLOC_CONF=pinned_max_round_threshold_mb:128,pinned_max_cached_size_mb:256
```

Example 15:

```bash
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:50,max_non_split_rounding_mb:30
```

Example 16:

```bash
export PYTORCH_NPU_ALLOC_CONF=release_lock_on_npumalloc:True
```

## Usage Constraints

- The `expandable_segments` feature must be used on Ascend HDK 23.0.0 or later.
- When either `max_split_size_mb` or `garbage_collection_threshold` is set to a non-default value, `expandable_segments` must be set to False.
<!-- npu="950,A3,910b" id10 -->
- The `page_size` feature requires Ascend HDK 25.0.RC1 or later and CANN Commercial Use 8.1.RC1 or later, and supports the following products:

    <!-- npu="910b" id1 -->
    - <term>Atlas A2 training products</term>
    <!-- end id1 -->
    <!-- npu="A3" id2 -->
    - <term>Atlas A3 training products</term>
    <!-- end id2 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950DT products</term>
    <!-- end id9 -->
- When the `page_size` feature is configured together with other features, only the `page_size` configuration takes effect. Notes on requesting memory are as follows:
    - When the requested memory is greater than 1 MB:
        - If `page_size` is configured, the memory allocation granularity is 1 GB.
        - If `page_size` is not configured, the memory allocation granularity is 2 MB.

    - When the requested memory is less than or equal to 1 MB: `page_size` does not take effect even if configured, and the memory allocation granularity is 2 MB.
<!-- end id10 -->
- The `pin_memory_expandable_segments` feature requires TorchNPU 7.3.0 or later, Ascend HDK 25.5.0 or later, and CANN Commercial Use 8.5.0 or later.
- The `pinned_use_background_threads` feature requires TorchNPU 26.0.0 or later and PyTorch 2.8.0 or later.
- Notes on using `pinned_mem_register` are as follows:
    - This feature requires TorchNPU 26.0.0 or later, Ascend HDK 26.0.RC1 or later, and CANN Commercial Use 8.5.0 or later.
    - This feature does not support being configured together with the `pin_memory_expandable_segments` feature.
- Notes on using `multi_stream_lazy_reclaim`:
    - This feature requires TorchNPU 7.3.0 or later.
    - This feature mainly solves system efficiency problems when the host side has a dispatch performance bottleneck in multi-stream scenarios. In single-stream or few-stream scenarios, or when the host is not the performance bottleneck, this feature provides limited benefit.
- The large\_segment\_size\_mb feature requires TorchNPU 26.1.0 and later versions and PyTorch 2.11.0 and later versions.
- The per\_process\_memory\_fraction feature requires TorchNPU 26.1.0 and later versions and PyTorch 2.10.0 and later versions.
- The throw\_on\_npumalloc\_oom feature requires TorchNPU 26.2.0 and later versions and PyTorch 2.13.0 and later versions.
- The pinned\_max\_round\_threshold\_mb and pinned\_max\_cached\_size\_mb features require TorchNPU 26.2.0 and later versions and PyTorch 2.13.0 and later versions. They apply only to the default (non-expandable) pinned memory allocator path. When configured together with pin\_memory\_expandable\_segments, these two thresholds do not take effect, and the framework outputs one warning message, but the process can still start normally. To enable these two thresholds, set pin\_memory\_expandable\_segments to False.
- Configuring caching allocator parameters through the [PYTORCH_ALLOC_CONF](PYTORCH_ALLOC_CONF.md) environment variable requires TorchNPU 26.2.0 and PyTorch 2.10.0 and later versions.

## Supported Products

<!-- npu="910" id3 -->
- <term>Atlas training products</term>
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas inference products</term>
<!-- end id6 -->
<!-- npu="950" id11 -->
- <term>Ascend 950DT products</term>
<!-- end id11 -->