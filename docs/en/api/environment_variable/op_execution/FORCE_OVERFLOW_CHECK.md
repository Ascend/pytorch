# FORCE\_OVERFLOW\_CHECK

## Function Description

This environment variable enables the overflow check switch in non-saturation mode (INF_NAN mode), which is used for asynchronous location of Inf/NaN problems during training without changing the floating-point computation mode. The default value of the environment variable is "0".

- When the environment variable value is "0": the overflow check switch is not enabled, and the behavior is the same as when it is not configured.
- When the environment variable value is "1": the overflow check switch is enabled.

When the environment variable is enabled, the overflow check APIs ([get\_npu\_overflow\_flag](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/en/custom_APIs/torch_npu-npu/(beta)torch_npu-npu-get_npu_overflow_flag.md), [npu\_check\_overflow](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/en/custom_APIs/torch_npu-npu/(beta)torch_npu-npu-utils-npu_check_overflow.md), [clear\_npu\_overflow\_flag](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/en/custom_APIs/torch_npu-npu/(beta)torch_npu-npu-clear_npu_overflow_flag.md)) are available in INF_NAN mode. You can use the overflow status flag to determine whether a value overflows, without moving gradients to the Host side to determine whether the value is Inf/NaN, avoiding synchronous waiting. This is suitable for overflow location scenarios that are sensitive to execution timing.

<!-- npu="A3,910b" id5 -->
The following is an example of using get\_npu\_overflow\_flag on <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> (other overflow check APIs are used in a similar way):

```python
import torch
import torch_npu
a = torch.tensor([65504.], dtype=torch.float16).npu()
b = a + a  # fp16 overflow; the result is inf in INF_NAN mode
torch_npu.npu.synchronize()
print(torch_npu.npu.utils.get_npu_overflow_flag())  # True, detected via the overflow status flag
```
<!-- end id5 -->

> [!NOTE]  
>
> <!-- npu="950" id4 -->
> - <term>Ascend 950DT products</term> do not support using APIs such as `get_npu_overflow_flag` to query the overflow status. When the environment variable switch is enabled on <term>Ascend 950DT products</term>, you need to confirm that the overflow check switch has taken effect through the `aclrtSetStreamOverflowSwitch`-related logs in plog (with `ASCEND_GLOBAL_LOG_LEVEL=1` configured).
> <!-- end id4 -->

This environment variable is provided by TorchNPU. PyTorch does not have a directly corresponding environment variable.

## Configuration Example

```bash
export FORCE_OVERFLOW_CHECK=1
```

## Usage Constraints

- The CANN version must be 9.1.0 or later. If the version does not meet the requirement, a WARNING log is printed and the environment variable is ignored.
- Only "0" or "1" is supported. Other values are regarded as not configured.
- It must be configured before the process starts. Changes made while the process is running do not take effect.
- After the overflow check switch is enabled, operator execution performance may be affected. It is recommended to configure it only in overflow/NaN problem location scenarios.

## Supported Products

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>
<!-- end id3 -->
