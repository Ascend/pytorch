# TORCH_HCCL\_BLOCKING\_WAIT

## Feature Description

When HCCL is used as the communication backend, this environment variable controls the synchronization mode (blocking or non-blocking) of `wait()` and `synchronize()` in `ProcessGroupHCCL`.

- `0` (default): disables blocking wait.
- `1`: enables blocking wait.

> [!NOTE]
>
> - The current version is also compatible with the old name `HCCL_BLOCKING_WAIT`.
> - When both `TORCH_HCCL_BLOCKING_WAIT` and `HCCL_BLOCKING_WAIT` are configured, `TORCH_HCCL_BLOCKING_WAIT` takes precedence.
> - When this environment variable is enabled, `wait()` and `synchronize()` return only after the current HCCL communication actually completes, reports an error, or times out on the host side.
> - When this environment variable is enabled, no watchdog thread is created.

## Configuration Example

The recommended configuration method is as follows:

```bash
export TORCH_HCCL_BLOCKING_WAIT=1
```

The configuration method compatible with the old name is as follows:

```bash
export HCCL_BLOCKING_WAIT=1
```

## Usage Constraints

- This environment variable takes effect only when HCCL is used as the communication backend.
- This environment variable is read during `ProcessGroupHCCL` initialization. If it is modified after the process group is created through `init_process_group` or `new_group`, the already created process group is not affected.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
- <term>Ascend 950DT</term>
