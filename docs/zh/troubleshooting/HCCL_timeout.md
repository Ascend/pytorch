# HCCL超时

## 问题现象

关键词“**EI0002**”

```ColdFusion
[W compiler_depend.ts:487] Warning: NPU warning, error code is 507048[Error]: 
[Error]: The execution of the internal task times out. 
        Rectify the fault based on the error information in the ascend log.
EI0002: [PID: 7277] 2024-11-28-11:59:30.645.648 The wait execution of the Notify register times out. Reason: The Notify register has not received the Notify record from remote rank [unknown].base information: [streamID:[2282987520], taskID[5], tag[AllReduce_90.90.92.240%enp189s0f0_60000_0_1732766162316022], AlgType(level 0-1-2):[fullmesh-ring-ring].] task information: [
there are(is) 1 abnormal device(s):
    Cluster Exception Location[IP/ID]:[90.90.92.240/1], Arrival Time:[Thu Nov 28 11:56:34 2024], Discoverer:[90.90.92.240/0], ExceptionType:[Heartbeat Lost Occurred], Possible Reason:1. Process has exited, 2. Network Disconnected
]
        Possible Cause: 1. An exception occurs during the execution on some NPUs in the cluster. As a result, collective communication operation failed.2. The execution speed on some NPU in the cluster is too slow to complete a communication operation within the timeout interval. (default 1800s, You can set the interval by using HCCL_EXEC_TIMEOUT.)3. The number of training samples of each NPU is inconsistent.4. Packet loss or other connectivity problems occur on the communication link.
```

## 故障根因

关键过程：模型多卡训练过程中，中断报错。

根本原因分析：其中一张卡异常退出，其他卡等待超时报错。

## 处理方法


<table><tbody><tr id="row133331920165614"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.1.1"><p id="p83339201562">Error Code</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.1.1 "><p id="p19428111123714">无</p>
</td>
</tr>
<tr id="row58261416152019"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.2.1"><p id="p78261916142014">故障事件名称</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.2.1 "><p id="p12416104333517">HCCL超时</p>
</td>
</tr>
<tr id="row1082711617201"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.3.1"><p id="p1782741619205">故障解释/可能原因</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.3.1 "><p id="p1960982933813">执行脚本存在错误</p>
</td>
</tr>
<tr id="row1474663022115"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.4.1"><p id="p774617303213">故障影响</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.4.1 "><p id="p35447463353">模型训练终止</p>
</td>
</tr>
<tr id="row19915122652114"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.5.1"><p id="p1791515262213">故障自处理模式</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.5.1 "><p id="p14790124723516">根据报错提示，检查脚本并修改</p>
</td>
</tr>
<tr id="row1356182417228"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.6.1"><p id="p175662413229">系统处理建议</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.6.1 "><p id="p164198119376">无需操作</p>
</td>
</tr>
</tbody>
</table>

