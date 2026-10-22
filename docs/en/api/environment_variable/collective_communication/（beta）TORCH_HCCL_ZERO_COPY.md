# \(beta\) TORCH\_HCCL\_ZERO\_COPY

> [!NOTICE]  
> This feature is still in the experimental stage. Use it with caution.

## Function Description

In training or online inference scenarios, this environment variable enables the on-chip zero-copy feature of collective communication, reducing the number of on-chip copies of communication operators during communication, improving collective communication efficiency, and reducing communication time. In computation-communication parallel scenarios, it also reduces the preemption of memory bandwidth during communication.

- 0: Disables the collective communication zero-copy feature.
- 1: Enables the collective communication zero-copy feature.

The default value is 0.

## Configuration Example

```bash
export TORCH_HCCL_ZERO_COPY=1
```

## Usage Constraints

- This environment variable depends on the TorchNPU virtual memory management feature. See [PYTORCH\_NPU\_ALLOC\_CONF](../memory_management/PYTORCH_NPU_ALLOC_CONF.md). The configuration must meet the following requirements:

    ```bash
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    ```

- This environment variable is not supported in TorchAir scenarios.
- For other constraints, see the "Zero-Copy Feature \> [Read Before Use](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/zero_copy_readme.md)" section in the *CANN HCCL Collective Communication Library*.

## Supported Products

<!-- npu="A3" id1 -->
<term>Atlas A3 training products</term>
<!-- end id1 -->
