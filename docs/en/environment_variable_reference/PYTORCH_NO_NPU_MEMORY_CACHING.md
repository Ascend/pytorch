# PYTORCH\_NO\_NPU\_MEMORY\_CACHING

## Feature Description

This environment variable configures whether to disable the memory reuse mechanism.

- When not configured or set to `0`, the memory reuse mechanism is enabled.
- When set to `1`, the memory reuse mechanism is disabled.

This environment variable is not configured by default.

After the memory reuse mechanism is disabled, each time memory is requested through the `aclrtMallocAlign32` or `aclrtMalloc` interface, it is immediately released back to the driver through the `aclrtFree` interface after its lifecycle ends.

> [!CAUTION]  
>
> - When the memory reuse mechanism is disabled, the `aclrtMalloc` and `aclrtFree` interfaces are used by default, and virtual memory is disabled by default.
> - You can disable the memory reuse mechanism for debugging purposes. After configuration, model performance may degrade, and the degradation may be significant in model scenarios with frequent memory allocation and release.

## Configuration Example

Example of disabling the memory reuse mechanism:

```bash
export PYTORCH_NO_NPU_MEMORY_CACHING=1
```

Example of re-enabling the memory reuse mechanism:

```bash
unset PYTORCH_NO_NPU_MEMORY_CACHING
# or
export PYTORCH_NO_NPU_MEMORY_CACHING=0
```

## Usage Constraints

If you need to use `torch_npu.npu.check_uce_in_memory`, this environment variable must be unconfigured, meaning the memory reuse mechanism is enabled.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
- <term>Ascend 950DT</term>
