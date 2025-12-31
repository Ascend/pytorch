# 用于梯度计算的变量被inplace操作

## 问题现象描述

回显信息中存在关键字“**one of the variables needed for gradient computation has been modified by an inplace operation**”，类似如下打印信息：

```ColdFusion
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

## 原因分析

调用`torch.autograd.backward`的过程中失败。

原地操作是指直接在原有张量上进行修改，而不创建新的副本。这样做会导致梯度无法正确计算，从而引发上述错误。

## 解决措施

根据日志信息找到报错的代码行，将原地操作改为非原地操作。例如：将x += 2改为y = x + 2。

