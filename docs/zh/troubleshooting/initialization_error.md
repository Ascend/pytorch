# 初始化报错

## 问题现象

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

## 故障根因

关键过程：初始化过程中报错。

根本原因分析：初始化时调用底层接口中间出现报错。

## 处理方法


<table><tbody><tr id="row133331920165614"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.1.1"><p id="p83339201562">Error Code</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.1.1 "><p id="p19428111123714">ERR00100</p>
</td>
</tr>
<tr id="row58261416152019"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.2.1"><p id="p78261916142014">故障事件名称</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.2.1 "><p id="p12416104333517">初始化失败</p>
</td>
</tr>
<tr id="row1082711617201"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.3.1"><p id="p1782741619205">故障解释/可能原因</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.3.1 "><p id="p435810129442">初始化时调用底层接口中间出现报错</p>
</td>
</tr>
<tr id="row1474663022115"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.4.1"><p id="p774617303213">故障影响</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.4.1 "><p id="p35447463353">脚本报错退出</p>
</td>
</tr>
<tr id="row19915122652114"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.5.1"><p id="p1791515262213">故障自处理模式</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.5.1 "><p id="p1560422614445">参考CANN文档<span id="ph124555416557">中“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/troubleshooting/troubleshooting_0062.html" target="_blank" rel="noopener noreferrer">问题定位思路</a>”章节中的指导</span>进行故障处理</p>
</td>
</tr>
<tr id="row1356182417228"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.6.1"><p id="p175662413229">系统处理建议</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.6.1 "><p id="p164198119376">无需操作</p>
</td>
</tr>
</tbody>
</table>

