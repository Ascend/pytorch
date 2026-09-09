# Automatic Core Binding

## Introduction

TorchNPU can enable coarse-grained or fine-grained core binding by setting the environment variable `CPU_AFFINITY_CONF`. This configuration avoids thread preemption, improves the cache hit rate, avoids memory access across NUMA (Non-Uniform Memory Access) nodes, reduces task scheduling overhead, and optimizes task execution efficiency.

The available core binding schemes are as follows:

- **Coarse-grained binding**: Binds all tasks to all CPU cores within the NPU service core binding range, preventing thread preemption between tasks on different cards.
- **Fine-grained binding**: Further optimization based on coarse-grained binding, anchoring TorchNPU hotspot threads (main thread, second-level pipeline thread, and so on) to fixed CPU cores within the NPU service core binding range. Specifically, the main thread is bound to the first CPU core in the core binding range, the second-level pipeline thread is bound to the second CPU core in the core binding range, and so on. Non-hotspot threads (such as dataloader threads) are bound to the remaining CPU cores in the range, isolated from hotspot threads to reduce the overhead of core switching.

    > [!NOTE]  
    > NPU service core binding range: When the core binding feature is enabled, the default core binding range for the service of each NPU card is the corresponding range obtained by evenly dividing the total number of CPU cores by the total number of NPU cards. For example, if the environment has 160 CPU cores and 8 NPU cards, the core binding range for the service of NPU card 0 is [0,19], that is, the first range after dividing into eight equal parts. The core binding range for the service of NPU card 1 is [20,39], and so on. In addition, you can specify the core binding range for the service of a particular card by adding parameters in the environment variable. For details, see [Usage Guide](#usage-guide).

**Figure 1**  Schematic diagram of thread core binding timing and policy design  
![figure](../figures/thread_core_binding_principle.png)

## Application Scenarios

This feature is recommended when the host takes a long time to dispatch tasks or when the service time varies greatly between cards.

## Usage Guide

Environment variable `CPU_AFFINITY_CONF=<mode>,force:<value0>,npu<value1>:<value2>-<value3>,npu_affine:<value4>`

1. `<mode>`: Required parameter, indicating the core binding mode.
    - 0 or not set: Indicates that the core binding feature is disabled.
    - 1: Indicates that coarse-grained binding is enabled.
    - 2: Indicates that fine-grained binding is enabled.

2. `force:<value0>`: Optional parameter, indicating whether to force core binding and skip conflict detection.
    - Not configuring this parameter (default) or setting it to 0: Keeps the original behavior. If the configured core binding range conflicts with the current CPU affinity of the process, the binding is skipped and a WARNING log is printed.
    - 1: Skips conflict detection and force-applies the core binding configuration. It applies to scenarios such as container core isolation. For example, when available cores are limited through Cgroup (Control Groups), the misjudgment of conflict detection caused by the Cgroup limit on the number of available CPU cores is avoided, which would otherwise invalidate the core binding configuration.

    > [!NOTE]
    >
    > `force:1` only skips conflict detection at the software level and cannot break the hard limits of the OS or Cgroup. The actual core binding result is still constrained by the CPU core range allowed by the operating system.

3. `npu<value1>:<value2>-<value3>`: Optional parameter, indicating a custom NPU service core binding range. The custom NPU service core binding range takes effect only when the core binding feature is enabled, that is, when mode is set to 1 or 2.
    - `npu<value1>:<value2>-<value3>` indicates that the "value1"-th card is bound to CPU cores in the closed interval from "value2" to "value3". For example, "npu0:0-2" indicates that the core binding range for the service threads of NPU card 0 is [0,2].
    - Multiple service core binding ranges can be configured for the same NPU card. For example, when the environment variable `CPU_AFFINITY_CONF=1,npu0:0-2,npu0:4-5` is set, the service core binding ranges of NPU card 0 are [0,2] and [4,5].
    - Service core binding ranges can be configured for multiple NPU cards. For example, when the environment variable `CPU_AFFINITY_CONF=1,npu0:0-0,npu2:1-2,npu2:4-4` is set, the service core binding range of NPU card 0 is changed to [0,0], that of NPU card 2 is changed to [1,2] and [4,4], and other NPU cards keep their original service core binding ranges.

4. `npu_affine:<value4>`: Optional parameter, indicating whether to enable NPU affinity binding.
    - 0 or not set: Indicates that the affinity binding feature is not enabled.
    - 1: Indicates that the affinity binding feature is enabled.

The core binding feature is disabled by default. If you need to improve performance through core binding, you are advised to use fine-grained binding.

> [!NOTE]  
>
>- The CPU core groups corresponding to NUMA nodes can be viewed using the `lscpu` command.
>- When binding cores, check whether the topology of the virtual machine is consistent with that of the physical machine. By default, the core group corresponding to npu0 or device 0 is NUMA0. However, virtual machine environments such as Docker may change the mapping relationship. You are advised to customize the core binding range based on the mapping relationship.
>- Before binding, the core binding range is checked. If any CPU core in the core binding range is non-affine, the thread is determined to already have affinity, and the core binding corresponding to this environment variable is not triggered (the default behavior when the `force` parameter is not configured). In container core isolation scenarios, this detection may misjudge. You can set `force:1` to skip the detection.
>- The optimization effect of core binding varies across different models. In some service scenarios, additional threads may exist, and thread preemption may instead cause performance degradation.
>- For user-defined threads, because child threads inherit the affinity of the parent thread, you are advised to manage the CPU affinity of child threads by calling `torch_npu.utils.set_thread_affinity` and `torch_npu.utils.reset_thread_affinity` before and after the location where child threads are spawned. For details, refer to the "[torch_npu.utils.set_thread_affinity](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/en/custom_APIs/torch_npu-utils/torch_npu-utils.set_thread_affinity.md)" section in *Custom API Reference* and the "[torch_npu.utils.reset_thread_affinity](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/en/custom_APIs/torch_npu-utils/torch_npu-utils.reset_thread_affinity.md)" section in *Custom API Reference*.
>- The affinity core binding range can be viewed using the `npu-smi info -t topo` command.

## Usage Examples

- Coarse-grained binding example:

    ```shell
    export CPU_AFFINITY_CONF=1
    ```

- Fine-grained binding example:

    ```shell
    export CPU_AFFINITY_CONF=2
    ```

- Custom NPU service core binding range example:

    For example, the core binding range for NPU card 0 is [0,1], for NPU card 1 is [2,5] and [7,8], for NPU card 3 is [9,9], and the core binding ranges for other NPU cards use the default settings. The configuration method is as follows:

    ```bash
    export CPU_AFFINITY_CONF=1,npu0:0-1,npu1:2-5,npu1:7-8,npu3:9-9
    ```

- NPU affinity binding example:

    ```shell
    export CPU_AFFINITY_CONF=1,npu_affine:1
    ```

- Forced core binding example in container core isolation scenarios:

    When the container limits the available cores through Cgroup, use `force:1` to skip conflict detection and force the core binding configuration to be applied:

    ```shell
    export CPU_AFFINITY_CONF=2,force:1,npu0:0-3
    ```

## Constraints

Affinity binding is supported only on <term>Atlas A2 training products</term>.
