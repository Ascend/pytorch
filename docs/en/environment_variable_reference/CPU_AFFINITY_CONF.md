# CPU_AFFINITY_CONF

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-08T10:23:17.694Z pushedAt=2026-07-08T10:47:16.878Z -->

## Feature Description

Ascend Extension for PyTorch enables coarse-grained or fine-grained core binding by setting the environment variable `CPU_AFFINITY_CONF`. This configuration prevents thread preemption, improves cache hit rates, avoids cross-NUMA (Non-Uniform Memory Access) node memory access, reduces task scheduling overhead, and optimizes task execution efficiency.

The available core binding schemes are as follows:

- **Coarse-grained binding**: Binds all tasks to all CPU cores within the NPU service core binding range, preventing thread preemption between tasks on different devices.
- **Fine-grained binding**: Further optimization based on coarse-grained binding, which anchors torch_npu hotspot threads (main thread, second-level pipeline thread, etc.) to fixed CPU cores within the NPU service core binding range. Specifically, the main thread is bound to the first CPU core in the core binding range, the second-level pipeline thread is bound to the second CPU core in the core binding range, and so on. Non-hotspot threads (such as dataloader threads) are bound to the remaining CPU cores in the range, isolated from hotspot threads, reducing the overhead of inter-core switching.

    > [!NOTE]  
    >
    > NPU service core binding range: When the core binding feature is enabled, the default core binding range for each NPU card's service is the corresponding segment after the total number of CPU cores is evenly divided by the total number of NPU cards. For example, assuming an environment has 160 CPU cores and 8 NPU cards, when core binding is enabled, the core binding range for the service of NPU card 0 is \[0,19\], i.e., the first segment after dividing into eight equal parts; the core binding range for the service of NPU card 1 is \[20,39\], and so on. In addition, users can also specify the core binding range for a particular card's service by adding parameters to the environment variable. See the parameter settings below for details.

Parameter configuration format: `CPU_AFFINITY_CONF=<mode>`,`npu<value1>:<value2>-<value3>`,`npu_affine:<value4>`

Parameter settings:

1. `<mode>`: Required parameter, indicating the core binding mode.
    - 0 or not set: indicates that the core binding function is disabled.
    - 1: indicates that coarse-grained binding is enabled.
    - 2: indicates that fine-grained binding is enabled.

2. `npu<value1>:<value2>-<value3>`: optional parameter, indicating a custom NPU service core binding range. The custom NPU service core binding range takes effect only when the core binding feature is enabled, that is, when mode is set to 1 or 2.
    - npu<value1\>:<value2\>-<value3\> indicates that the "value1"-th card is bound to CPU cores in the closed interval from "value2" to "value3". For example, "npu0:0-2" indicates that the core binding range for the service threads of NPU card 0 is \[0,2\].
    - Supports modifying the service core binding range for some NPU cards. For example, when setting the environment variable CPU\_AFFINITY\_CONF=1,npu0:0-0, the service core binding range of NPU card 0 is changed to \[0,0\], while NPU card 1 retains its original service core binding range.

3. `npu_affine:<value4>`: Optional parameter, indicating whether to enable NPU affinity binding.
    - 0 or not set: Indicates that the affinity binding feature is not enabled.
    - 1: Indicates that the affinity binding feature is enabled.

The core binding feature is disabled by default. If core binding is needed to improve performance, fine-grained binding is recommended.

> [!NOTE]
>
> - The CPU core group corresponding to a NUMA node can be viewed using the **lscpu** command.
> - When binding cores, check whether the topology of the virtual machine matches that of the physical machine. By default, the core group corresponding to npu0 or Device 0 is NUMA0; however, virtualized environments such as Docker may alter the mapping relationship. It is recommended to customize the core binding range based on the actual mapping relationship.
> - The core binding range is checked before binding. If any CPU core within the binding range is non-affinity, the thread is determined to already have affinity, and the core binding triggered by this environment variable will not take effect.
> - The degree of optimization from core binding varies across different models. In some service scenarios, additional threads may exist, and thread preemption may instead lead to performance degradation.
> - For user-defined threads, since child threads inherit the affinity of the parent thread, it is recommended to manage the CPU affinity of child threads by calling `torch_npu.utils.set_thread_affinity` and `torch_npu.utils.reset_thread_affinity` before and after the location where child threads are spawned. For details, refer to the "[torch_npu.utils.set_thread_affinity](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/zh/custom_APIs/torch_npu-utils/torch_npu-utils.set_thread_affinity.md)" section in *Ascend Extension for PyTorch Custom API Reference* and the "[torch_npu.utils.reset_thread_affinity](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/zh/custom_APIs/torch_npu-utils/torch_npu-utils.reset_thread_affinity.md)" section in *Ascend Extension for PyTorch Custom API Reference*.
> - The affinity core binding range can be viewed using the **npu-smi info -t topo** command.

## Configuration Examples

Example 1: Coarse-grained Binding

```bash
export CPU_AFFINITY_CONF=1
```

Example 2: Fine-grained Binding

```bash
export CPU_AFFINITY_CONF=2
```

Example 3: Custom NPU Service Core Binding Range

For example, the core binding range for NPU card 0 is \[0,1\], for NPU card 1 is \[2,5\], for NPU card 3 is \[6,6\], and the core binding ranges for other NPU cards use the default settings. The configuration method is as follows:

```bash
export CPU_AFFINITY_CONF=1,npu0:0-1,npu1:2-5,npu3:6-6
```

Example 4: NPU affinity binding

```bash
export CPU_AFFINITY_CONF=1,npu_affine:1
```

## Usage Constraints

Affinity binding is only supported on <term>Atlas A2 training products</term>.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
