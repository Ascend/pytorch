# HCCL超时

## 问题现象描述

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

## 原因分析

由于其中一张卡异常退出，导致其他卡在等待超时后报错，模型多卡训练发生中断。

## 解决措施

根据错误码EI0002，可参见《HCCL集合通信库用户指南》中的“[执行通信操作超时（EI0002）](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/hccl/hcclug/hcclug_000031.html)”章节指导进行故障处理。

