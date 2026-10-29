# P2P\_HCCL\_BUFFSIZE

## Function Description

This environment variable configures whether point-to-point communication (torch.distributed.isend, torch.distributed.irecv, and torch.distributed.batch\_isend\_irecv) uses an independent communicator, as well as the buffer size of the independent communicator.

- When set to 0: The independent point-to-point communicator is not used, and point-to-point communication reuses the default communicator.
- When set to a value greater than or equal to 1: The independent point-to-point communicator is used, and the buffer size is the configured value.

The unit is MB, and the default value is 20.

When the independent communicator function for point-to-point communication is enabled, each communicator will additionally occupy a buffer of size 2\*P2P\_HCCL\_BUFFSIZE. If there are many communicators in the cluster network, this buffer usage will increase, potentially affecting the normal storage of model data. In this scenario, you can use this environment variable to reduce the buffer size occupied by point-to-point communicators. If the model data volume of the service is small but the point-to-point communication data volume is large, you can use this environment variable to increase the buffer size occupied by point-to-point communicators, thereby improving point-to-point communication efficiency.

## Configuration Example

```bash
export P2P_HCCL_BUFFSIZE=20
```

## Usage Constraints

- The memory requested by this environment variable is exclusive to HCCL and cannot be reused by other service memory.
- Each communicator additionally occupies memory of size "2\*P2P\_HCCL\_BUFFSIZE", which is used for sending and receiving data respectively.
- This resource is managed at the granularity of communicators, with each domain exclusively occupying a set of memory of size "2\*P2P\_HCCL\_BUFFSIZE".
- In TorchNPU 7.1.0, this environment variable is configured to 20 MB by default. If an OOM error occurs after the upgrade, you can set this environment variable to 0 on the model side.
- If an independent communicator was not previously created for P2P, configuring this environment variable will independently create a P2P communicator. If there is a long interval between send and recv dispatch on the model side, a timeout may occur. In this case, you need to set HCCL\_CONNECT\_TIMEOUT to a longer value. The recommended value is 600s, and the specific value must be adjusted based on the actual model script. For details about HCCL\_CONNECT\_TIMEOUT, see the "[HCCL\_CONNECT\_TIMEOUT](https://gitcode.com/cann/hccl/blob/9.2.0/docs/en/user_guide/hccl_env/HCCL_CONNECT_TIMEOUT.md)" section in the *CANN HCCL Library*.

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT products</term>
<!-- end id4 -->
