# Parameter Verification Failed for Operator Call

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-12T08:23:18.612Z pushedAt=2026-06-12T11:22:41.041Z -->

## Symptom

The output message contains the keyword "**Expected all tensors to be on the same device. Expected NPU tensor, please check whether the input tensor device is correct.**", similar to the following print information:

```ColdFusion
Traceback (most recent call last):
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/testing/_internal/common_utils.py", line 2388, in wrapper
    method(*args, **kwargs)
  File "npu/test_fault_mode.py", line 114, in test_param_verification
    torch.add(torch.rand(2), torch.rand(2).npu())
RuntimeError: Expected all tensors to be on the same device. Expected NPU tensor, please check whether the input tensor device is correct.
[ERROR] 2024-08-18-22:37:51 (PID:53741, Device:0, RankID:-1) ERR01002 OPS invalid type
```

## Possible Cause

When using torch-related APIs to operate on tensors, the error code **ERR01002** is printed.

The device types of the input tensors are inconsistent, causing the operator to fail to be called normally. The tensors need to have their devices all set to CPU or all set to NPU.

## Solution

Check whether the attributes such as device and dtype of the input tensors are correct. If the device types are inconsistent, modify them to the same device.

For example, change:

```python
torch.add(torch.rand(2), torch.rand(2).npu())
```

to:

```python
torch.add(torch.rand(2).npu(), torch.rand(2).npu())
```
