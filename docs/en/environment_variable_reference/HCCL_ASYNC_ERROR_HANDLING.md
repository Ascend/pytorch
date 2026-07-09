# HCCL\_ASYNC\_ERROR\_HANDLING

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:03:31.045Z pushedAt=2026-06-16T03:14:22.183Z -->

## Function Description

When using HCCL as the communication backend, this environment variable controls whether to enable asynchronous error handling.

- 0: Disables asynchronous error handling.
- 1: Enables asynchronous error handling.

The default value is **0** when the PyTorch version is 1.11.0, and **1** when the PyTorch version is 2.1.0 or later.

> [!NOTE]  
> In the current version, when asynchronous handling is enabled, the process will be terminated if a CQE error occurs. For other error messages, only on-screen information prompts are displayed, and the process will not be terminated.

## Configuration Example

```bash
export HCCL_ASYNC_ERROR_HANDLING=1
```

## Usage Constraints

When enabling asynchronous error handling through this environment variable, to better identify the cause of HCCL timeouts, it is recommended to set the timeout parameter of `new_group` and `init_process_group` to a value greater than the time configured by the `HCCL_CONNECT_TIMEOUT` and `HCCL_EXEC_TIMEOUT` environment variables. For details about `HCCL_CONNECT_TIMEOUT`, see the "[HCCL_CONNECT_TIMEOUT](https://www.hiascend.com/document/detail/en/canncommercial/900/maintenref/envvar/envref_07_0077.html)" section in the *CANN Environment Variable Reference*. For details about `HCCL_EXEC_TIMEOUT`, see the "[HCCL_EXEC_TIMEOUT](https://www.hiascend.com/document/detail/en/canncommercial/900/maintenref/envvar/envref_07_0078.html)" section in the *CANN Environment Variable Reference*.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
