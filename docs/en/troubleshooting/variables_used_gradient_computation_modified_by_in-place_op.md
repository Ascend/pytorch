# Variables Used for Gradient Computation Modified by In-place Operations

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-12T08:25:38.657Z pushedAt=2026-06-12T11:22:41.064Z -->

## Symptom

The output contains the keyword "**one of the variables needed for gradient computation has been modified by an inplace operation**", similar to the following print information:

```text
ERROR: test_autograd_backward (__main__.TestMode)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/testing/_internal/common_utils.py", line 2388, in wrapper
    method(*args, **kwargs)
  File "npu/test_fault_mode.py", line 159, in test_autograd_backward
    torch.autograd.grad(d2.sum(), a)
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/autograd/__init__.py", line 394, in grad
    result = Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [5]], which is output 0 of AddBackward0, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
```

## Possible Cause

The call of `torch.autograd.backward` failed.

An in-place operation refers to modifying the original tensor directly without creating a new copy. Doing so prevents the gradient from being computed correctly, thus triggering the above error.

## Solution

Locate the erroneous code line based on the log information, and change the in-place operation to a non-in-place operation. For example: change `x += 2` to `y = x + 2`.
