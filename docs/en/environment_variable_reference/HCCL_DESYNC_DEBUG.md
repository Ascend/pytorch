# HCCL\_DESYNC\_DEBUG

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:03:42.299Z pushedAt=2026-06-16T03:14:22.189Z -->

## Function Description

When using HCCL as the communication backend, this environment variable controls whether to perform communication timeout analysis.

- 0: Disable communication timeout analysis.
- 1: Enable communication timeout analysis.

Default value: 0.

> [!NOTE]  
>
> - In the current version, only the timeout analysis result is printed, and the process is not terminated.
> - When the cluster networking scale is large, if this environment variable is enabled, the training process may hang abnormally.

## Configuration Example

```bash
export HCCL_DESYNC_DEBUG=1
```

## Usage Constraints

When the PyTorch version is 1.11.0, this environment variable must be used together with [HCCL\_ASYNC\_ERROR\_HANDLING](HCCL_ASYNC_ERROR_HANDLING.md). That is, if `HCCL_DESYNC_DEBUG` is set to `1`, `HCCL_ASYNC_ERROR_HANDLING` must also be set to `1`.

## Supported Models

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
