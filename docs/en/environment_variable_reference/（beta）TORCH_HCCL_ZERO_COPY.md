# \(Beta\) TORCH\_HCCL\_ZERO\_COPY

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:02:48.806Z pushedAt=2026-06-16T03:14:22.131Z -->

> [!NOTICE]  
> This feature is still in the experimental stage. Use it with caution.

## Feature Description

In training or online inference scenarios, this environment variable can be used to enable the on-chip zero-copy feature for collective communication, reducing the number of on-chip copies performed by communication operators during communication, improving collective communication efficiency, and lowering communication latency. Additionally, in computation-communication overlap scenarios, it reduces contention for video memory bandwidth during communication.

- 0: Disable the on-chip zero-copy feature for collective communication.
- 1: Enable the on-chip zero-copy feature for collective communication.

The default value is 0.

## Configuration Example

```bash
export TORCH_HCCL_ZERO_COPY=1
```

## Usage Constraints

- This environment variable depends on the virtual memory management feature of Ascend Extension for PyTorch. See [PYTORCH\_NPU\_ALLOC\_CONF](PYTORCH_NPU_ALLOC_CONF.md). The configuration must meet the following requirements:

    ```bash
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    ```

- This environment variable is not supported in PyTorch graph mode (TorchAir) scenarios.
- For other constraints, see "Zero-Copy Function" > "[Before You Start](https://www.hiascend.com/document/detail/en/canncommercial/900/API/hcclug/hcclcpp_07_0053.html)" section in the *CANN HCCL Library*.

## Supported Products

<term>Atlas A3 training series</term>
