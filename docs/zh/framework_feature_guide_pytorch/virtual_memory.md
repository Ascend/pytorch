# 虚拟内存

## 简介

虚拟内存管理机制主要是将地址和内存的概念分离。通过配置环境变量PYTORCH\_NPU\_ALLOC\_CONF或修改torch.npu.memory.\_set\_allocator\_settings接口“expandable\_segments“属性值，使PyTorch管理虚拟内存和物理内存的映射，并允许多次申请使用连续内存，通过构建可扩展的内存段，动态调整内存块的大小，可以减少内存碎片的产生。对于有较多内存碎片的模型，可以降低设备内存占用率。

如[图1](#虚拟地址与物理内存映射)所示，通过底层提供的接口，可以率先预留大块的虚拟地址空间作为一个内存块。用户不断申请内存时，可以得到在连续地址上的内存块，分别与不同地址的物理内存映射，释放内存时，连续地址的内存块可以合并成为一个内存块。

**图 1**  虚拟地址与物理内存映射  <a id="虚拟地址与物理内存映射"></a>  
![](figures/mapping_between_virtual_addresses_and_physical_memory.png)

## 使用场景

训练过程中出现内存溢出（Out Of Memory，OOM）时，或想降低模型内存占用率的情况下，可以考虑使用该特性。

## 使用指导

可选择如下任一方式：

-   设置环境变量PYTORCH\_NPU\_ALLOC\_CONF=expandable\_segments:<value\>，此环境变量使用详情请参考《环境变量参考》中的“PYTORCH\_NPU\_ALLOC\_CONF”章节。
-   修改torch.npu.memory.\_set\_allocator\_settings（“expandable\_segments: <value\>”）接口中的“expandable\_segments“属性值。

    value可以取值为True或False。默认为False。

    -   设置为True时，此设置将指示缓存分配器创建特定的内存块分配，这些内存块后续可以扩展，以便能更好地处理内存使用中频繁变更使用内存大小的情况。

    -   设置为False时，关闭内存池扩展段功能，使用原有的内存申请方式。

## 使用样例

-   PYTORCH\_NPU\_ALLOC\_CONF使用样例如下所示。

    开启虚拟内存机制：

    ```shell
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    ```

    关闭虚拟内存机制：

    ```shell
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    ```

-   torch.npu.memory.\_set\_allocator\_settings使用样例如下所示。

    开启虚拟内存机制：

    ```Python
    torch.npu.memory._set_allocator_settings("expandable_segments:True")
    ```

    关闭虚拟内存机制：

    ```Python
    torch.npu.memory._set_allocator_settings("expandable_segments:False")
    ```

## 约束说明

expandable\_segments特性需在Ascend HDK 23.0.0及以上版本上使用。

torch.npu.memory.\_set\_allocator\_settings仅支持在PyTorch2.6.0及以上版本使用。

torch.npu.memory.\_set\_allocator\_settings当前仅支持修改expandable\_segments属性。

