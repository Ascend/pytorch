# TORCH\_HCCL\_STATUS\_SAVE\_INTERVAL

## Function Description

When HCCL is used as the communication backend, this environment variable configures the interval for saving HCCL process group status.

- The default value of this environment variable is 2, in seconds (s).
- If a value less than or equal to 0 is configured, it is reset to 2.

> [!NOTE]
>
> This environment variable is read and cached on the first call.

This environment variable is provided by TorchNPU. PyTorch has no directly corresponding environment variable.

## Configuration Example

```bash
export TORCH_HCCL_STATUS_SAVE_INTERVAL=5
```

## Usage Constraints

It takes effect only when `TORCH_HCCL_STATUS_SAVE_ENABLE=1`.

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT products</term>
<!-- end id4 -->
