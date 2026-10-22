# Initialization Error

## Symptom

Keyword "**at\_npu::native::AclSetCompileopt**"

```text
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

## Cause Analysis

An error occurs when the underlying interface is called during initialization, and the error code "ERR00100" is printed.

## Solution

See the "[Process Interruption Problem Locating](https://gitcode.com/cann/docs/blob/9.2.0/docs/en/troubleshooting/process_interrupt_analysis.md)" section in *CANN Fault Handling* for troubleshooting.
