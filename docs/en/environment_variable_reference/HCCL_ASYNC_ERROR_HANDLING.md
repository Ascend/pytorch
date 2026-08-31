# HCCL\_ASYNC\_ERROR\_HANDLING

## Feature Description

When using HCCL as the communication backend, this environment variable controls whether to enable asynchronous error handling.

- 0: Disables asynchronous error handling.
- 1: Enables asynchronous error handling.

The default value is 0 when the PyTorch version is 1.11.0, and 1 when the PyTorch version is 2.1.0 or later.

> [!NOTE]  
> In the current version, when asynchronous error handling is enabled, the process is terminated if a CQE error occurs. For other errors, only a prompt message is displayed, and the process is not terminated.

## Configuration Example

```bash
export HCCL_ASYNC_ERROR_HANDLING=1
```

## Usage Constraints

When enabling asynchronous error handling through this environment variable, to better identify the cause of HCCL timeouts, you are advised to set the timeout of the `new_group` and `init_process_group` parameters to a value greater than the time configured by the `HCCL_CONNECT_TIMEOUT` and `HCCL_EXEC_TIMEOUT` environment variables. For details about `HCCL_CONNECT_TIMEOUT`, see the "HCCL\_CONNECT\_TIMEOUT" section in the *CANN HCCL Collective Communication Library*. For details about `HCCL_EXEC_TIMEOUT`, see the "HCCL\_EXEC\_TIMEOUT" section in the *CANN HCCL Collective Communication Library*.
<!-- "[HCCL\_CONNECT\_TIMEOUT](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_CONNECT_TIMEOUT.md)" -->
<!-- "[HCCL\_EXEC\_TIMEOUT](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md)" -->

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
