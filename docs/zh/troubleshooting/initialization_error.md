# 初始化报错

## 问题现象描述

关键词"**at\_npu::native::AclSetCompileopt**"

```ColdFusion
File "/home/HwHiAiUser/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/serialization.py", line 359, in _privateuse1_deserialize
 return getattr(obj, backend_name)(device_index)
 File "/home/HwHiAiUser/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/utils/backend_registration.py", line 225, in wrap_storage_to
 untyped_storage = torch.UntypedStorage(
 File "/home/HwHiAiUser/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch_npu/npu/__init__.py", line 214, in _lazy_init
 torch_npu._C._npu_init()
 RuntimeError: Initialize:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:217 NPU function error: at_npu::native::AclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, precision_mode), error code is 500001
 [ERROR] 2025-01-15-20:59:48 (PID:6038, Device:0, RankID:-1) ERR00100 PTA call acl api failed
 [Error]: The internal ACL of the system is incorrect.
 Rectify the fault based on the error information in the ascend log.
```

## 原因分析

初始化时调用底层接口的过程中出现报错，并打印错误码“ERR00100”。

## 解决措施

参考《CANN 故障处理》中的“[进程中断问题定位思路](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/troubleshooting/troubleshooting_0062.html)”章节进行故障处理。