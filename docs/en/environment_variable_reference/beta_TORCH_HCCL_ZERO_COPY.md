# \(beta\) TORCH\_HCCL\_ZERO\_COPY

> [!NOTICE]  
> This feature is still in the experimental stage. Use it with caution.

## Feature Description

In training or online inference scenarios, this environment variable can be used to enable the on-chip zero-copy feature for collective communication, reducing the number of on-chip copies performed by communication operators during communication, improving collective communication efficiency, and lowering communication latency. Additionally, in computation-communication overlap scenarios, it reduces contention for on-device memory bandwidth during communication.

- 0: Disable the on-chip zero-copy feature for collective communication.
- 1: Enable the on-chip zero-copy feature for collective communication.

The default value is 0.

## Configuration Example

```bash
export TORCH_HCCL_ZERO_COPY=1
```

## Usage Constraints

- This environment variable depends on the virtual memory management feature of TorchNPU. See [PYTORCH\_NPU\_ALLOC\_CONF](PYTORCH_NPU_ALLOC_CONF.md). The configuration must meet the following requirements:

    ```bash
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    ```

- This environment variable is not supported in TorchAir scenarios.
- For other constraints, see "Zero-Copy Function" > "[Before You Start](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/commlib/hcclug/docs/en/api_ref/comm_mgr_c/zero_copy_readme.md)" section in the *CANN HCCL Library*.

## Supported Products

<term>Atlas A3 training products</term>
