# Operator Call Error

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-12T08:23:58.396Z pushedAt=2026-06-12T11:22:41.050Z -->

## Symptom

Keyword **Cannot find bin of op ...**

```output
Traceback (most recent call last):
  File "/home/HwHiAiUser/workspace/qwen2.5-Math-deepseek-R1.py", line 38, in <module>
    generated_ids = model.generate(
  File "/home/HwHiAiUser/.local/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/home/HwHiAiUser/.local/lib/python3.10/site-packages/transformers/generation/utils.py", line 1575, in generate
    result = self._sample(
  File "/home/HwHiAiUser/.local/lib/python3.10/site-packages/transformers/generation/utils.py", line 2690, in _sample
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
RuntimeError: call aclnnArange failed, detail:EZ9999: Inner Error!
EZ9999: [PID: 775] 2025-02-16-12:18:25.258.321 Cannot find bin of op Range, integral key 0/1/|int64/ND/int64/ND/int64/ND/int64/ND/.
        TraceBack (most recent call last):
       Cannot find binary for op Range.
       Kernel GetWorkspace failed. opType: 9
       ArangeAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.
[ERROR] 2025-02-16-12:18:25 (PID:775, Device:0, RankID:-1) ERR01100 OPS call acl api failed
```

## Possible Cause

An operator call error occurs, and the error code **ERR01100** is printed. Possible causes:

- The corresponding Kernels or ops package is not installed.
- No matching operator binary file is found.

## Solution

1. Check whether the corresponding kernels or ops package has been installed.
2. Check whether the input data type of the operator is supported. If not, use a supported data type. For details, refer to the [Ascend IR Operator Specifications](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/aolapi/operatorlist_00094.html) section in the *CANN Operator Library*.
