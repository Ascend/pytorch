# PYTORCH\_NPU\_ALLOC\_CONF

## 功能描述

通过此环境变量可控制缓存分配器行为。配置此环境变量会改变内存占用量，可能造成性能波动。

缓存分配器会根据申请内存的大小使用不同内存池，小于1MB使用小块内存池，反之使用大块内存池；虚拟内存特性下，大块内存池申请的物理内存粒度（segment\_size\_mb）默认为20MB，小块内存池默认为2MB（不可配置）；大模型场景下小块内存池内存使用通常较少，因此部分环境配置项（page\_size、segment\_size\_mb）只作用于大块内存池。

可选参数：

-   max\_split\_size\_mb:<value\>，内存块允许切分上限。

    大于设定值的内存块在使用过程中不会进行切分，这有助于减少内存碎片。此选项主要应用于当模型由于OOM（Out of Memory，内存不足）而中断，并出现大量非活动的切分内存块场景的优化。<value\>默认值为无限大，单位MB，即所有大小的内存块都可以切分，最小设置值大于20MB。

-   garbage\_collection\_threshold:<value\>，垃圾回收阈值。

    主动回收未使用的NPU内存块。在设置value阈值（例如0.8）后，如果NPU内存容量使用超过阈值（即分配给NPU应用程序的总内存的80%），缓存分配器将开始回收NPU内存块，优先释放最先申请和长时间未复用的内存块，避免释放积极复用的内存块。其中<value\>取值范围为\(0.0,1.0\)。默认不开启该功能。垃圾回收阈值需与内存因子配合使用，内存因子可参考《[Ascend Extension for PyTorch 自定义API参考](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/overview.md)》的“torch\_npu.npu.set\_per\_process\_memory\_fraction”。

-   expandable\_segments:<value\>，使能内存池扩展段功能，即虚拟内存特性。

    默认为False。如果设置为True，此设置将指示缓存分配器创建特定的内存块分配，这些内存块后续可以扩展，以便能更好地处理内存使用中频繁变更使用内存大小的情况。如果设置为False，关闭内存池扩展段功能，使用原有的内存申请方式。

-   base\_addr\_aligned\_kb:<value\>，内存基地址对齐粒度。

    取值范围为0\~16，设置值需为整数，单位KB，默认值为16。仅在expandable\_segments设置为true的时候生效。若此参数配置为16，在申请大块内存（大于等于2MB）的时候，会尽量保持申请内存的基地址16KB对齐；若配置为0时，申请内存的基地址512B对齐。

-   page\_size:<value\>，设置申请大页内存的大小。

    取值仅支持1GB，参考[配置示例](#配置示例)。内存申请粒度为1GB，不足1GB的向上对齐到1GB，始终为1GB的整数倍。在虚拟内存特性下，该配置项只作用于缓存分配器的大块内存池。

    未配置该选项时，内存申请粒度为2MB，如需申请1GB的大页内存，会占用1024/2=512个页表；设置该选项后，内存申请粒度为1GB，1GB大页内存只占用1个页表，能有效降低页表数量，有效扩大TLB（Translation Lookaside Buffer）缓存的地址范围，从而提升离散访问的性能。

    TLB是昇腾AI处理器中用于高速缓存的硬件模块，用于存储最近使用的虚拟地址到物理地址的映射。

-   segment\_size\_mb:<value\>，虚拟内存特性下，设置物理内存的申请粒度。

    取值范围20\~512，设置值需为整数，单位MB，不配置时默认为20。仅在expandable\_segments设置为true的时候生效，改配置项只作用于缓存分配器的大块内存池。与page\_size同时配置时，page\_size优先生效。
    增大segment\_size\_mb可以减少内存申请及内存映射接口的调用次数，从而提升内存申请效率，但也可能带来更多的内存碎片。因此，在内存使用极限的场景下，请谨慎调大此值。

-   roundup\_power2\_divisions:<value\> 或 roundup\_power2\_divisions:\[<size1\>:<value1\>,<size2\>:<value2\>,...\]，将请求的分配大小向上舍入到最近的2的幂次分段，从而更高效地复用内存块。

    不配置时分配大小会以512字节为单位向上对齐，这对较小的分配尺寸效果良好；对于较大的、尺寸相近的分配请求，这种策略可能效率低下。因为每个请求会被分配到不同大小的内存块中，导致这些内存块难以被复用，继而产生大量未被充分复用的内存块，浪费内存容量。

    支持两种配置方式：

    -   **单一值**：为每个内存设置相同的分段数量，例如配置为“4“。
    -   **键值对数组**：为每个2的幂区间单独设置分段数量。例如配置为“\[256:1,512:2,1024:4,\>:8\]“时，表示为256MB以下的所有分配设置1个分段，256MB到512MB之间的分配设置2个分段，512MB到1GB之间的分配设置4个分段，以及更大的分配设置8个分段。

-   pinned\_use\_background\_threads:<value\>，是否启用后台线程来处理events。

    默认值为False，不启用后台线程。当设置为True时，启用后台线程，在后台线程执行查询和处理events操作，减少主线程的阻塞时间。

-   pin\_memory\_expandable\_segments:<value\>，使能pin_memory内存池扩展段功能，即虚拟内存特性。

    默认为False。如果设置为True，此设置将指示pin_memory缓存分配器内存池物理内存申请粒度为20MB（不可配置），创建的内存块后续可以扩展，以便能更好地处理内存使用中频繁变更内存大小的情况，同时pin_memory内存块计数相关统计指标不参与统计（默认值：0）。如果设置为False，关闭pin_memory内存池扩展段功能，使用原有的内存申请方式。

-   pinned\_mem\_register:<value\>，设置pin_memory内存是否启用host register功能。

    默认为False。如果设置为True，此设置将指示pin_memory内存启用host register功能，将pin_memory内存映射注册为Device可访问的内存地址。如果设置为False，关闭host register功能。


> [!NOTE]  
> 用户使用Ascend Extension for PyTorch 6.0.RC3及之后版本配套的驱动（Ascend HDK 24.1.RC3及之后），开启虚拟内存特性时，可以使用单进程多卡特性；用户使用Ascend Extension for PyTorch 6.0.RC3之前版本配套的驱动（Ascend HDK 24.1.RC3之前版本），开启虚拟内存特性时，不能使用单进程多卡特性。

## 配置示例<a id="配置示例"></a> 

示例一：

```
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
```

示例二：

```
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,base_addr_aligned_kb:16
```

示例三：

```
export PYTORCH_NPU_ALLOC_CONF=page_size:1g
```

示例四：

```
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,segment_size_mb:40
```

示例五：

-   单一值示例：

    ```
    export PYTORCH_NPU_ALLOC_CONF="roundup_power2_divisions:4"
    ```

-   键值对数组示例：

    ```
    export PYTORCH_NPU_ALLOC_CONF="roundup_power2_divisions:[256:1,512:2,1024:4,>:8]"
    ```

示例六：

```
export PYTORCH_NPU_ALLOC_CONF=pinned_use_background_threads:True
```


示例七：

```
export PYTORCH_NPU_ALLOC_CONF=pin_memory_expandable_segments:True
```


示例八：

```bash
export PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True
```

## 使用约束

-   expandable\_segments特性需在Ascend HDK 23.0.0及以上版本上使用。
-   max\_split\_size\_mb和garbage\_collection\_threshold中任意一个为非默认值时，expandable\_segments必须设置为False。
-   page\_size特性要求在Ascend HDK 25.0.RC1及以上版本、CANN商发8.1.RC1及以上版本使用，支持如下产品：
    -   <term>Atlas A2 训练系列产品</term>
    -   <term>Atlas A3 训练系列产品</term>

-   page\_size特性与其他特性不支持同时配置，且申请内存注意事项如下：
    -   当申请内存大于1M时：
        -   若配置page\_size，内存申请粒度为1GB。
        -   若未配置page\_size，内存申请粒度为2MB。

    -   当申请内存小于等于1MB时：配置page\_size也不生效，内存申请粒度为2MB。
-   pin_memory_expandable_segments特性要求最低Ascend Extension for PyTorch 7.3.0之后版本、Ascend HDK 25.5.0及以上版本、CANN商发8.5.0及以上版本使用。
-   pinned_mem_register使用注意事项如下：
    -   特性要求Ascend Extension for PyTorch 26.0.0及以上版本、Ascend HDK 25.5.0及以上版本、CANN商发8.5.0及以上版本使用。
    -   与pin_memory_expandable_segments特性不支持同时配置。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 推理系列产品</term>

