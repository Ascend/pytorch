# 通信算子传入非连续tensor

## 问题现象

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

## 故障根因

关键过程：启动的分布式任务报该错误。

根本原因分析：通信算子传入了非连续tensor。

## 处理方法

根据日志信息找到报错的代码行，检查输入数据的连续性，通过`.contiguous()`将非连续tensor转换为连续tensor。


<table><tbody><tr id="row133331920165614"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.1.1"><p id="p83339201562">Error Code</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.1.1 "><p id="p19428111123714">ERR02002</p>
</td>
</tr>
<tr id="row58261416152019"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.2.1"><p id="p78261916142014">故障事件名称</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.2.1 "><p id="p1812603173">通信算子传入非连续tensor</p>
</td>
</tr>
<tr id="row1082711617201"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.3.1"><p id="p1782741619205">故障解释/可能原因</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.3.1 "><p id="p15427111113718">代码脚本问题</p>
</td>
</tr>
<tr id="row1474663022115"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.4.1"><p id="p774617303213">故障影响</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.4.1 "><p id="p15426111163718">通信算子失败</p>
</td>
</tr>
<tr id="row19915122652114"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.5.1"><p id="p1791515262213">故障自处理模式</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.5.1 "><p id="p142311119376">检查代码，保证通信算子传入的tensor是连续的</p>
</td>
</tr>
<tr id="row1356182417228"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.6.1"><p id="p175662413229">系统处理建议</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.6.1 "><p id="p164198119376">无需操作</p>
</td>
</tr>
</tbody>
</table>

