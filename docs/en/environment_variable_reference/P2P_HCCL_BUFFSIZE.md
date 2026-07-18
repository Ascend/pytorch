# P2P\_HCCL\_BUFFSIZE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:05:20.540Z pushedAt=2026-06-16T03:14:22.322Z -->

## Function Description

This environment variable configures whether to enable point-to-point communication (torch.distributed.isend, torch.distributed.irecv, and torch.distributed.batch\_isend\_irecv) and use the independent communication domain function.

- When configured to 0: Disables point-to-point communication and uses the independent communication domain function.
- When configured to a value greater than or equal to 1: Enables point-to-point communication using the independent communication domain function, and the buffer size is the configured value.

The unit is MB, and the default value is 20.

When the independent communication domain function for point-to-point communication is enabled, each communication domain will additionally occupy a buffer of size 2\*P2P\_HCCL\_BUFFSIZE. If there are many communication domains in the cluster network, this buffer usage will increase, potentially affecting the normal storage of model data. In this scenario, you can use this environment variable to reduce the buffer size occupied by point-to-point communication domains. If the model data volume of the service is small but the point-to-point communication data volume is large, you can use this environment variable to increase the buffer size occupied by point-to-point communication domains, thereby improving point-to-point communication efficiency.

## Configuration Example

```bash
export P2P_HCCL_BUFFSIZE=20
```

## Usage Constraints

- The memory requested by this environment variable is exclusive to HCCL and cannot be reused by other service memory.
- Each communication domain additionally occupies memory of size "2\*P2P\_HCCL\_BUFFSIZE", which is used for sending and receiving data respectively.
- This resource is managed at the granularity of communication domains, with each domain exclusively occupying a set of memory of size "2\*P2P\_HCCL\_BUFFSIZE".
- In Ascend Extension for PyTorch 7.1.0, this environment variable is configured to 20 MB by default. If an OOM error occurs after the upgrade, you can set this environment variable to 0 on the model side.
- If an independent communication domain was not previously created for P2P, configuring this environment variable will independently create a P2P communication domain. If there is a long interval between send and recv dispatch on the model side, a timeout may occur. In this case, you need to set HCCL\_CONNECT\_TIMEOUT to a longer value. The recommended value is 600s, and the specific value must be adjusted based on the actual model script. For details about HCCL\_CONNECT\_TIMEOUT, see the "[HCCL\_CONNECT\_TIMEOUT](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0077.html)" section in the *CANN Description of Environment Variables*.

## Supported Models

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
