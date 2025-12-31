# 调用算子参数校验失败

## 问题现象描述

回显信息中存在关键字“**Expected all tensors to be on the same device. Expected NPU tensor, please check whether the input tensor device is correct.**”，类似如下打印信息：

```ColdFusion
Traceback (most recent call last):
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/testing/_internal/common_utils.py", line 2388, in wrapper
    method(*args, **kwargs)
  File "npu/test_fault_mode.py", line 114, in test_param_verification
    torch.add(torch.rand(2), torch.rand(2).npu())
RuntimeError: Expected all tensors to be on the same device. Expected NPU tensor, please check whether the input tensor device is correct.
[ERROR] 2024-08-18-22:37:51 (PID:53741, Device:0, RankID:-1) ERR01002 OPS invalid type
```

## 原因分析

使用torch相关API对tensor操作时，打印错误码“ERR01002”。

输入的tensors的device类型不一致，导致算子不能正常调用，需要tensor的device同时为CPU或NPU。

## 解决措施

检查传入的tensor的device、dtype等属性是否正确。如果device类型不一致，需要修改为同一device。

比如将：
```
torch.add(torch.rand(2), torch.rand(2).npu())
```
改为：
```
torch.add(torch.rand(2).npu(), torch.rand(2).npu())
```
