# 调用算子参数校验失败

## 问题现象

回显信息中存在关键字“**Expected all tensors to be on the same device. Expected NPU tensor, please check whether the input tensor device is correct.**”，类似如下屏显信息：

```ColdFusion
Traceback (most recent call last):
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/testing/_internal/common_utils.py", line 2388, in wrapper
    method(*args, **kwargs)
  File "npu/test_fault_mode.py", line 114, in test_param_verification
    torch.add(torch.rand(2), torch.rand(2).npu())
RuntimeError: Expected all tensors to be on the same device. Expected NPU tensor, please check whether the input tensor device is correct.
[ERROR] 2024-08-18-22:37:51 (PID:53741, Device:0, RankID:-1) ERR01002 OPS invalid type
```

## 故障根因

关键过程：使用torch相关API对tensor操作。

根本原因分析：输入的tensors的device类型不一致，需要tensor的device同时为CPU或NPU。

## 处理方法


<table><tbody><tr id="row9346937267"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.1.1"><p id="p173461137163">Error Code</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.1.1 "><p id="p173460371465">ERR01002</p>
</td>
</tr>
<tr id="row2346163713620"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.2.1"><p id="p734613376614">故障事件名称</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.2.1 "><p id="p23469379614">调用算子参数校验失败</p>
</td>
</tr>
<tr id="row1834683719619"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.3.1"><p id="p1634618371464">故障解释/可能原因</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.3.1 "><p id="p11346837562">传入的参数信息有误</p>
</td>
</tr>
<tr id="row183465371265"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.4.1"><p id="p203466373619">故障影响</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.4.1 "><p id="p2346237465">算子不能正常调用</p>
</td>
</tr>
<tr id="row1134610376612"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.5.1"><p id="p73460376610">故障自处理模式</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.5.1 "><p id="p7671021193018">检查传入的tensor的device、dtype等属性是否正确。如果device类型不一致，需要修改为同一device。</p>
<p id="p650163661211">比如将：</p>
<pre class="screen" id="screen678312621415">torch.add(torch.rand(2), torch.rand(2).npu())</pre>
<p id="p5727123130">改为：</p>
<pre class="screen" id="screen19954125119131">torch.add(torch.rand(2).npu(), torch.rand(2).npu())</pre>
</td>
</tr>
<tr id="row23461371565"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.6.1"><p id="p33467371613">系统处理建议</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.6.1 "><p id="p63475379617">无需操作</p>
</td>
</tr>
</tbody>
</table>

