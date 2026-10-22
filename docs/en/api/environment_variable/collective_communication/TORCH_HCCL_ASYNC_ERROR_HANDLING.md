# TORCH\_HCCL\_ASYNC\_ERROR\_HANDLING

## Function Description

When using HCCL as the communication backend, this environment variable controls whether to enable asynchronous error handling.

- 0: Disables asynchronous error handling.
- 1: Enables asynchronous error handling.

The default value is 0 when the PyTorch version is 1.11.0, and 1 when the PyTorch version is 2.1.0 or later.

> [!NOTE]  
>
> - The current version is also compatible with the old name `HCCL_ASYNC_ERROR_HANDLING`.
> - When both `TORCH_HCCL_ASYNC_ERROR_HANDLING` and `HCCL_ASYNC_ERROR_HANDLING` are configured, `TORCH_HCCL_ASYNC_ERROR_HANDLING` takes precedence.
> - In the current version, when asynchronous processing is enabled, if an ERROR CQE error occurs, the process terminates. For other error information, only an on-screen message is displayed, and the process is not terminated.

## Configuration Example

The recommended configuration method is as follows:

```bash
export TORCH_HCCL_ASYNC_ERROR_HANDLING=1
```

The configuration method compatible with the old name is as follows:

```bash
export HCCL_ASYNC_ERROR_HANDLING=1
```

## Usage Constraints

When enabling asynchronous error handling through this environment variable, to better identify the cause of HCCL timeouts, you are advised to set the timeout of the `new_group` and `init_process_group` parameters to a value greater than the time configured by the `HCCL_CONNECT_TIMEOUT` and `HCCL_EXEC_TIMEOUT` environment variables. For details about `HCCL_CONNECT_TIMEOUT`, see the "[HCCL\_CONNECT\_TIMEOUT](https://gitcode.com/cann/hccl/blob/9.2.0/docs/en/user_guide/hccl_env/HCCL_CONNECT_TIMEOUT.md)" section in the *CANN HCCL Collective Communication Library*. For details about `HCCL_EXEC_TIMEOUT`, see the "[HCCL\_EXEC\_TIMEOUT](https://gitcode.com/cann/hccl/blob/9.2.0/docs/en/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md)" section in the *CANN HCCL Collective Communication Library*.

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
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>
<!-- end id4 -->
