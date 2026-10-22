# INDUCTOR\_ASCEND\_LOG\_LEVEL

## Function Description

This environment variable configures the log level of the Inductor module, controlling the verbosity of log output.

- Default value `WARNING`: outputs warning information.
- Set to `DEBUG`: most detailed debugging information.
- Set to `INFO`: general information.
- Set to `ERROR`: error information.
- Set to `CRITICAL`: critical error information.

> [!NOTE]
>
> This environment variable is read and cached when the `torch_npu._inductor` module is initialized; changes made during runtime do not take effect.

PyTorch configures the log level through `torch._inductor.config.log_level`; TorchNPU provides the related configuration through this environment variable.

## Configuration Example

```bash
export INDUCTOR_ASCEND_LOG_LEVEL=DEBUG
```

## Usage Constraints

None

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="950" id2 -->
- <term>Ascend 950DT products</term>
<!-- end id2 -->
