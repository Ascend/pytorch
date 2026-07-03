# CPU\_AFFINITY\_CONF

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:03:19.890Z pushedAt=2026-06-16T03:14:22.171Z -->

## Feature Description

Ascend Extension for PyTorch can enable coarse/fine-grained core binding by setting the environment variable `CPU_AFFINITY_CONF`. This configuration prevents inter-thread preemption, improves cache hit rates, avoids cross-NUMA (Non-Uniform Memory Access) node memory access, reduces task scheduling overhead, and optimizes task execution efficiency.

The available core binding schemes are as follows:

- **Coarse-grained binding**: Binds all tasks to all CPU cores within the NPU service core binding range, preventing thread preemption between tasks on different cards.
- **Fine-grained binding**: Further optimization based on coarse-grained binding, anchoring torch_npu hotspot threads (main thread, second-level pipeline thread, etc.) to fixed CPU cores within the NPU service core binding range. Specifically, the main thread is bound to the first CPU core in the core binding range, the second-level pipeline thread is bound to the second CPU core in the core binding range, and so on. Non-hotspot threads (such as dataloader threads) are bound to the remaining CPU cores in the range, isolated from hotspot threads, reducing inter-core switching overhead.

    > [!NOTE]  
    >
    > NPU service core binding range: When the core binding feature is enabled, the default core binding range for each NPU card's service is the corresponding range after the total number of CPU cores is equally divided by the total number of NPU cards. For example, if the environment has 160 CPU cores and 8 NPU cards, when core binding is enabled, the core binding range for the service of NPU card 0 is \[0,19\], that is, the first range after the eight-way split, the core binding range for the service of NPU card 1 is \[20,39\], and so on. In addition, users can also specify the core binding range for a specific card's service by adding parameters in the environment variable. For details, see the parameter settings below.

Parameter configuration format: `CPU_AFFINITY_CONF=<mode>,force:<value0>,npu<value1>:<value2>-<value3>,npu_affine:<value4>`

Parameter settings:

1. `<mode>`: Required parameter, indicating the core binding mode.

    - 0 or not set: Indicates the core binding function is disabled.

    - 1: Indicates coarse-grained binding is enabled.

    - 2: Indicates fine-grained binding is enabled.

2. `force:<value0>`: optional parameter, indicating whether to force core binding and skip conflict detection.

    - Not configured (default) or configured as 0: maintains the original behavior. When the configured core binding range conflicts with the process's current CPU affinity, core binding is skipped and a WARNING log is output.

    - 1: skips conflict detection and forces the app to apply the core binding configuration. This is suitable for scenarios such as container core isolation, for example, when available cores are restricted via Cgroup (Control Groups), preventing conflict detection misjudgment caused by Cgroup's limitation on available CPU cores, which would otherwise cause the core binding configuration to fail.

    > [!NOTE]  
    >
    > `force:1` only skips software-level conflict detection and cannot break through OS/Cgroup hard limits. The actual core binding result is still constrained by the CPU core range allowed by the operating system.

3. `npu<value1>:<value2>-<value3>`: optional parameter, indicating a custom NPU service core binding range. The custom NPU service core binding range takes effect only when the core binding feature is enabled, that is, when mode is configured as 1 or 2.

    - npu<value1\>:<value2\>-<value3\> indicates that the "value1"-th card is bound to the CPU cores in the closed interval from "value2" to "value3". For example, "npu0:0-2" indicates that the core binding range for the service threads of NPU card 0 is \[0,2\].
    - Configuring multiple service core binding ranges for the same NPU card is supported. For example, when the environment variable `CPU_AFFINITY_CONF=1,npu0:0-2,npu0:4-5` is set, the service core binding ranges for NPU card 0 are \[0,2\] and \[4,5\].
    - Configuring service core binding ranges for multiple NPU cards is supported. For example, when the environment variable CPU\_AFFINITY\_CONF=1,npu0:0-0,npu2:1-2,npu2:4-4 is set, the service core binding range for NPU card 0 is modified to \[0,0\], the service core binding ranges for NPU card 2 are modified to \[1,2\] and \[4,4\], and other NPU cards retain their original service core binding ranges.

4. `npu_affine:<value4>`: optional parameter, indicating whether to enable NPU affinity binding.
    - 0 or not set: indicates that the affinity binding function is not enabled.
    - 1: Indicates that the affinity core binding feature is enabled.

The core binding feature is disabled by default. If you need to improve performance through core binding, fine-grained binding is recommended.

> [!NOTE]
>
> - The CPU core groups corresponding to NUMA nodes can be viewed using the `lscpu` command.
> - When binding cores, check whether the topology of the virtual machine matches that of the physical machine. By default, the core group corresponding to npu0 or Device 0 is NUMA0; however, virtualized environments such as Docker may change the mapping relationship. It is recommended to customize the core binding range based on the mapping relationship.
> - Before binding cores, the core binding range is detected. If any CPU core within the core binding range is non-affine, the thread is determined to already have affinity, and the core binding corresponding to this environment variable will not be triggered (the default behavior when the `force` parameter is not configured). In container core isolation scenarios, this detection may cause misjudgment, which can be avoided by setting `force:1` to skip the detection.
> - The degree of optimization from core binding varies across different models. In some service scenarios, additional threads may exist, and thread preemption may lead to performance degradation instead.
> - For user-defined threads, since child threads inherit the affinity of the parent thread, it is recommended to manage the CPU affinity of child threads by calling torch\_npu.utils.set\_thread\_affinity and torch\_npu.utils.reset\_thread\_affinity before and after the location where child threads are created. For details, refer to the "[torch\_npu.utils.set\_thread\_affinity](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-utils/torch_npu-utils.set_thread_affinity.md)" section in the *Ascend Extension for PyTorch Custom API Reference* and the "[torch\_npu.utils.reset\_thread\_affinity](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-utils/torch_npu-utils.reset_thread_affinity.md)" section in the *Ascend Extension for PyTorch Custom API Reference*.
> - The affinity core binding range can be viewed using the **npu-smi info -t topo** command.

## Configuration Examples

Example 1: Coarse-grained binding

```bash
export CPU_AFFINITY_CONF=1
```

Example 2: Fine-grained binding

```bash
export CPU_AFFINITY_CONF=2
```

Example 3: Custom NPU service core binding range

For example, the core binding range for NPU card 0 is \[0,1\], for NPU card 1 is \[2,5\] and \[7,8\], for NPU card 3 is \[9,9\], and the core binding ranges for other NPU cards use the default settings. The configuration method is as follows:

```bash
export CPU_AFFINITY_CONF=1,npu0:0-1,npu1:2-5,npu1:7-8,npu3:9-9
```

Example 4: NPU affinity binding

```bash
export CPU_AFFINITY_CONF=1,npu_affine:1
```

Example 5: Forced core binding in container core isolation scenarios

When a container limits available cores through Cgroup, use `force:1` to skip conflict detection and force the app's core binding configuration:

```bash
export CPU_AFFINITY_CONF=2,force:1,npu0:0-3
```

## Usage Constraints

Affinity binding only supports <term>Atlas A2 training products</term>.

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
