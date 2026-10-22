# Non-contiguous Tensor Passed to a Communication Operator

## Symptom

The command output contains the keyword "**RuntimeError: Tensors must be contiguous**", similar to the following printed information:

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

## Cause Analysis

Because a non-contiguous tensor is passed to the communication operator, an error is reported when the distributed task is started, and the error code "ERR02002" is printed.

## Solution

There may be a problem in the code script. Locate the code line that reports the error based on the log information and check the contiguity of the input data. It is recommended to add the `.contiguous()` method before calling the communication operator, for example: `input_tensor = input_tensor.contiguous()`, to ensure that the tensor passed to the communication operator is contiguous.
