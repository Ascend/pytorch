# HCCL Timeout

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-12T08:23:26.820Z pushedAt=2026-06-12T11:22:41.044Z -->

## Symptom

Keyword **EI0002**

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

## Possible Cause

An abnormal exit of one card causes other cards to report an error after a timeout while waiting, interrupting multi-card model training.

## Solution

Based on the error code **EI0002**, for details, see the [Communication Operation Timeout (EI0002)](https://www.hiascend.com/document/detail/en/canncommercial/83RC1/hccl/hcclug/hcclug_000031.html) section in the *HCCL User Guide* for troubleshooting guidance.
