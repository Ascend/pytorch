# Non-contiguous Tensors Passed to Communication Operators

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-12T08:20:44.005Z pushedAt=2026-06-12T11:22:41.013Z -->

## Symptom

The output information contains the keyword **RuntimeError: Tensors must be contiguous**, with similar print as follows:

```text
Traceback (most recent call last):
  File "distributed/_mode_cases/error_discontinuous_tensor.py", line 21, in <module>
    discontinuous_tensor()
  File "distributed/_mode_cases/error_discontinuous_tensor.py", line 18, in discontinuous_tensor
    dist.all_reduce(input)
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
    return func(*args, **kwargs)
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 2050, in all_reduce
    work = group.allreduce([tensor], opts)
RuntimeError: Tensors must be contiguous
[ERROR] 2024-08-18-22:15:47 (PID:23232, Device:0, RankID:0) ERR02002 DIST invalid type
```

## Possible Cause

The error occurs because communication operators receive non-contiguous tensors, causing the distributed task to fail at startup and print the error code **ERR02002**.

## Solution

The code script may have issues.
Locate the erroneous code line based on the log information and check the contiguity of the input data. It is recommended to add the `.contiguous()` method (for example: `input_tensor = input_tensor.contiguous()`) before calling communication operators to ensure that the tensors passed to the communication operators are contiguous.
