# CPU_AFFINITY_CONF

## Function Description

You can enable coarse-grained and fine-grained CPU binding for TorchNPU by setting the environment variable `CPU_AFFINITY_CONF`. This configuration prevents thread preemption, improves the cache hit rate, avoids memory access across NUMA (Non-Uniform Memory Access) nodes, reduces task scheduling overhead, and optimizes task execution efficiency.

The following binding schemes are available:

- **Coarse-grained binding**: Binds all tasks to all CPU cores in the CPU binding range of the NPU service, preventing thread preemption between tasks on different NPU cards.
- **Fine-grained binding**: Further optimizes coarse-grained binding by anchoring TorchNPU hot threads (for example, the main thread and second-level pipeline threads) to fixed CPU cores in the CPU binding range of the NPU service. That is, the main thread is bound to the first CPU core in the binding range, the second-level pipeline threads are bound to the second CPU core, and so on. Non-hot threads (for example, dataloader threads) are bound to the remaining CPU cores in the range and are isolated from hot threads, which reduces inter-core switching overhead.

    > [!NOTE]  
    >
    > CPU binding range of the NPU service: When CPU binding is enabled, the default binding range of each NPU card service is the range obtained by evenly dividing the total number of CPU cores by the total number of NPU cards. For example, suppose the environment has 160 CPU cores and 8 NPU cards. When binding is enabled, the binding range of NPU card 0 is [0,19], that is, the first range after dividing the cores into eight parts. The binding range of NPU card 1 is [20,39], and so on. In addition, you can add parameters to the environment variable to specify the binding range of a card service. For details, see Parameter settings.

Configuration format: `CPU_AFFINITY_CONF=<mode>,force:<value0>,npu<value1>:<value2>-<value3>,npu_affine:<value4>`

Parameter settings:

1. `<mode>`: Required parameter. Specifies the binding mode.
    - `0` or not set: Disables CPU binding.
    - `1`: Enables coarse-grained binding.
    - `2`: Enables fine-grained binding.

2. `force:<value0>`: Optional parameter. Specifies whether to force binding and skip conflict detection.
    - Not set (default) or set to `0`: Keeps the original behavior. When the configured binding range conflicts with the current CPU affinity of the process, the binding is skipped and a WARNING log is generated.
    - `1`: Skips conflict detection and forcibly applies the binding configuration. It applies to scenarios such as container core isolation, where the available cores are limited by Cgroup (Control Groups). This prevents conflict detection misjudgment caused by the Cgroup limitation on the number of available CPU cores, which would otherwise invalidate the binding configuration.

    > [!NOTE]  
    >
    > `force:1` only skips software-level conflict detection. It cannot break through the hard limit of the OS or Cgroup. The actual binding result is still constrained by the CPU core range allowed by the operating system.

3. `npu<value1>:<value2>-<value3>`: Optional parameter. Specifies a custom CPU binding range for the NPU service. The custom binding range takes effect only when the binding feature is enabled, that is, when `mode` is set to `1` or `2`.
    - `npu<value1>:<value2>-<value3>` binds the threads of card `value1` to the CPU cores in the closed interval from `value2` to `value3`. For example, `npu0:0-2` indicates that the binding range of the service threads of NPU card 0 is [0,2].
    - You can configure multiple binding ranges for one NPU card. For example, if you set the environment variable `CPU_AFFINITY_CONF=1,npu0:0-2,npu0:4-5`, the binding range of NPU card 0 is [0,2] and [4,5].
    - You can configure binding ranges for multiple NPU cards. For example, if you set the environment variable `CPU_AFFINITY_CONF=1,npu0:0-0,npu2:1-2,npu2:4-4`, the binding range of NPU card 0 is changed to [0,0], the binding range of NPU card 2 is changed to [1,2] and [4,4], and the binding ranges of other NPU cards remain unchanged.

4. `npu_affine:<value4>`: Optional parameter. Specifies whether to enable NPU affinity binding.
    - `0` or not set: Disables affinity binding.
    - `1`: Enables affinity binding.

CPU binding is disabled by default. If you need to improve performance through CPU binding, you are advised to use fine-grained binding.

> [!NOTE]
>
> - Run the `lscpu` command to view the CPU core groups corresponding to NUMA nodes.
> - When binding CPU cores, ensure that the topology of the virtual machine is consistent with that of the physical machine. By default, the core group corresponding to `npu0` or Device 0 is NUMA0. However, virtual machine environments such as Docker may change the mapping. You are advised to customize the binding range based on the actual mapping.
> - Before binding, the binding range is checked. If any CPU core in the binding range is non-affine to the thread, the thread is considered to already have affinity, and the binding corresponding to the environment variable is not triggered (the default behavior when the `force` parameter is not set). In the container core isolation scenario, this check may misjudge the situation. You can set `force:1` to skip the check.
> - The optimization effect of binding varies across models. Some service scenarios may have additional threads, in which case thread preemption may instead degrade performance.
> - For user-defined threads, because child threads inherit the affinity of the main thread, you are advised to call `torch_npu.utils.set_thread_affinity` and `torch_npu.utils.reset_thread_affinity` around the position where child threads are created to manage the CPU affinity of the child threads. For details, see the "[torch_npu.utils.set_thread_affinity](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/en/custom_APIs/torch_npu-utils/torch_npu-utils.set_thread_affinity.md)" and "[torch_npu.utils.reset_thread_affinity](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/en/custom_APIs/torch_npu-utils/torch_npu-utils.reset_thread_affinity.md)" sections in *Custom APIs*.
> - Run the `npu-smi info -t topo` command to view the affinity binding range.

## Configuration Examples

Example 1: Coarse-grained binding

```bash
export CPU_AFFINITY_CONF=1
```

Example 2: Fine-grained binding

```bash
export CPU_AFFINITY_CONF=2
```

Example 3: Customizing the CPU binding range of the NPU service

For example, the binding range of NPU card 0 is [0,1], the binding range of NPU card 1 is [2,5] and [7,8], the binding range of NPU card 3 is [9,9], and the binding ranges of other NPU cards use the default settings. Set the binding ranges as follows:

```bash
export CPU_AFFINITY_CONF=1,npu0:0-1,npu1:2-5,npu1:7-8,npu3:9-9
```

Example 4: NPU affinity binding

```bash
export CPU_AFFINITY_CONF=1,npu_affine:1
```

Example 5: Forcing binding in the container core isolation scenario

When the container limits the available cores through Cgroup, set `force:1` to skip conflict detection and forcibly apply the binding configuration:

```bash
export CPU_AFFINITY_CONF=2,force:1,npu0:0-3
```

## Usage Constraints

Affinity binding is supported only on Atlas A2 training products.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950DT</term>
