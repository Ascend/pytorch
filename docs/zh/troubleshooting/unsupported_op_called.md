# 调用不支持的算子

## 问题现象

关键字"Warning: CAUTION: The operator 'xxx' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications. \(function npu\_cpu\_fallback\)"

```ColdFusion
[W compiler_depend.ts:51] Warning: CAUTION: The operator 'aten::linalg_lstsq.out' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications. (function npu_cpu_fallback)
Traceback (most recent call last):
  File "temp_1.py", line 4, in <module>
    torch.linalg.lstsq(torch.randn(1, 3, 3).npu(), torch.randn(2, 3, 3).npu())
RuntimeError: _copy_from_and_resize now only support copy with same size!
[ERROR] 2024-11-28-11:37:15 (PID:6547, Device:0, RankID:-1) ERR01007 OPS feature not supported
```

## 故障根因

关键过程：当模型运行时，屏显信息会打印该警告。

根本原因分析：一些算子在NPU上还不支持。

## 处理方法


<table><tbody><tr id="row133331920165614"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.1.1"><p id="p83339201562">Error Code</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.1.1 "><p id="p19428111123714">ERR01007</p>
</td>
</tr>
<tr id="row58261416152019"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.2.1"><p id="p78261916142014">故障事件名称</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.2.1 "><p id="p12416104333517">调用不支持的算子</p>
</td>
</tr>
<tr id="row1082711617201"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.3.1"><p id="p1782741619205">故障解释/可能原因</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.3.1 "><p id="p19444174583517">一些算子在NPU上还不支持</p>
</td>
</tr>
<tr id="row1474663022115"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.4.1"><p id="p774617303213">故障影响</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.4.1 "><p id="p35447463353">可能影响性能</p>
</td>
</tr>
<tr id="row19915122652114"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.5.1"><p id="p1791515262213">故障自处理模式</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.5.1 "><p id="p14790124723516">如果只是触发告警无报错，在不考虑改善性能的情况下可不处理；其他情况，请改用torch其他可替换并支持的接口</p>
</td>
</tr>
<tr id="row1356182417228"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.6.1"><p id="p175662413229">系统处理建议</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.6.1 "><p id="p164198119376">无需操作</p>
</td>
</tr>
</tbody>
</table>

