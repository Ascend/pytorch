# TORCH\_HCCL\_DESYNC\_DEBUG

## Function Description

When using HCCL as the communication backend, this environment variable controls whether to perform communication timeout analysis.

- 0: Disable communication timeout analysis.
- 1: Enable communication timeout analysis.

Default value: 0.

> [!NOTE]
>
> - The current version is also compatible with the old name `HCCL_DESYNC_DEBUG`.
> - When both `TORCH_HCCL_DESYNC_DEBUG` and `HCCL_DESYNC_DEBUG` are configured, `TORCH_HCCL_DESYNC_DEBUG` takes precedence.
> - In the current version, only the timeout analysis result is printed, and the process is not terminated.
> - When the cluster networking scale is large, if this environment variable is enabled, the training process may hang abnormally.

## Configuration Example

The recommended configuration method is as follows:

```bash
export TORCH_HCCL_DESYNC_DEBUG=1
```

The configuration method compatible with the old name is as follows:

```bash
export HCCL_DESYNC_DEBUG=1
```

## Usage Constraints

When the PyTorch version is 1.11.0, this environment variable must be used together with [TORCH\_HCCL\_ASYNC\_ERROR\_HANDLING](TORCH_HCCL_ASYNC_ERROR_HANDLING.md). That is, if TORCH\_HCCL\_DESYNC\_DEBUG is set to 1, TORCH\_HCCL\_ASYNC\_ERROR\_HANDLING must also be set to 1.

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
<!-- npu="950" id5 -->
- <term>Ascend 950DT products</term>
<!-- end id5 -->
