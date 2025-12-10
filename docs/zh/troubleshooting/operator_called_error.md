# 算子调用报错

## 问题现象

关键词"**Cannot find bin of op ...**"

```ColdFusion
Traceback (most recent call last):
  File "/home/HwHiAiUser/workspace/qwen2.5-Math-deepseek-R1.py", line 38, in <module>
    generated_ids = model.generate(
  File "/home/HwHiAiUser/.local/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/home/HwHiAiUser/.local/lib/python3.10/site-packages/transformers/generation/utils.py", line 1575, in generate
    result = self._sample(
  File "/home/HwHiAiUser/.local/lib/python3.10/site-packages/transformers/generation/utils.py", line 2690, in _sample
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
RuntimeError: call aclnnArange failed, detail:EZ9999: Inner Error!
EZ9999: [PID: 775] 2025-02-16-12:18:25.258.321 Cannot find bin of op Range, integral key 0/1/|int64/ND/int64/ND/int64/ND/int64/ND/.
        TraceBack (most recent call last):
       Cannot find binary for op Range.
       Kernel GetWorkspace failed. opType: 9
       ArangeAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.
[ERROR] 2025-02-16-12:18:25 (PID:775, Device:0, RankID:-1) ERR01100 OPS call acl api failed
```

## 故障根因

关键过程：算子调用报错。

根本原因分析：没有找到调用算子的二进制文件。

## 处理方法


<table><tbody><tr id="row133331920165614"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.1.1"><p id="p83339201562">Error Code</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.1.1 "><p id="p19428111123714">ERR01100</p>
</td>
</tr>
<tr id="row58261416152019"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.2.1"><p id="p78261916142014">故障事件名称</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.2.1 "><p id="p12416104333517">算子调用报错</p>
</td>
</tr>
<tr id="row1082711617201"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.3.1"><p id="p1782741619205">故障解释/可能原因</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.3.1 "><ol id="ol141563489380"><li>没有安装配套Kernels包</li><li>没有匹配的算子二进制文件</li></ol>
</td>
</tr>
<tr id="row1474663022115"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.4.1"><p id="p774617303213">故障影响</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.4.1 "><p id="p35447463353">算子调用报错退出</p>
</td>
</tr>
<tr id="row19915122652114"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.5.1"><p id="p1791515262213">故障自处理模式</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.5.1 "><ol id="ol12398152216402"><li>检查安装配套的Kernels包</li><li>查看算子输入数据类型是否支持，如果不支持请使用支持的数据类型，可参考<span id="ph124555416557">《CANN 算子库接口参考》中“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/aolapi/operatorlist_00094.html" target="_blank" rel="noopener noreferrer">Ascend IR算子规格说明</a>”章节</span>进行查看</li></ol>
</td>
</tr>
<tr id="row1356182417228"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.6.1"><p id="p175662413229">系统处理建议</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.6.1 "><p id="p164198119376">无需操作</p>
</td>
</tr>
</tbody>
</table>

