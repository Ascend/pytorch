# 用于梯度计算的变量被inplace操作

## 问题现象

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

## 故障根因

关键过程：调用`torch.autograd.backward`的过程中失败。

根本原因分析：原地操作是指直接在原有张量上进行修改，而不创建新的副本。这样做会导致梯度无法正确计算，从而引发上述错误。

## 处理方法

根据日志信息找到报错的代码行，将原地操作改为非原地操作。例如：将x += 2改为y = x + 2。


<table><tbody><tr id="row133331920165614"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.1.1"><p id="p83339201562">Error Code</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.1.1 "><p id="p19428111123714">无</p>
</td>
</tr>
<tr id="row58261416152019"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.2.1"><p id="p78261916142014">故障事件名称</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.2.1 "><p id="p12416104333517">用于梯度计算的变量被inplace操作</p>
</td>
</tr>
<tr id="row1082711617201"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.3.1"><p id="p1782741619205">故障解释/可能原因</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.3.1 "><p id="p19444174583517">代码脚本问题</p>
</td>
</tr>
<tr id="row1474663022115"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.4.1"><p id="p774617303213">故障影响</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.4.1 "><p id="p35447463353">反向传播无法正常计算</p>
</td>
</tr>
<tr id="row19915122652114"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.5.1"><p id="p1791515262213">故障自处理模式</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.5.1 "><p id="p14790124723516">将原地操作的算子改为非原地操作</p>
</td>
</tr>
<tr id="row1356182417228"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.6.1"><p id="p175662413229">系统处理建议</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.6.1 "><p id="p164198119376">无需操作</p>
</td>
</tr>
</tbody>
</table>

