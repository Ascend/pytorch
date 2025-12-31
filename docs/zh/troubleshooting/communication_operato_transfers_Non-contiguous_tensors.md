# 通信算子传入非连续tensor

## 问题现象描述

回显信息中存在关键字“**RuntimeError: Tensors must be contiguous**”，类似如下打印信息：

```ColdFusion
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

## 原因分析

由于通信算子传入非连续tensor，导致启动分布式任务时报错，并打印错误码“ERR02002”。



## 解决措施

代码脚本可能存在问题。
根据日志信息找到报错的代码行，检查输入数据的连续性，通过`.contiguous()`将非连续tensor转换为连续tensor，保证通信算子传入的tensor是连续的。


