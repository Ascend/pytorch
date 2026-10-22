# TORCHNPU\_PRECOMPILE\_THREADS

## Function Description

This environment variable configures the number of precompile threads of the `torch_npu` Inductor. Precompile threads are used to speed up parallel compilation tasks during graph compilation.

- Default value `max(os.cpu_count() // max(compile_threads, 2), 32)`: the number of precompile threads is not less than 32, where `compile_threads` is controlled by `TORCHINDUCTOR_COMPILE_THREADS`.
- Set to a positive integer: uses the specified number of threads; when set to an invalid value (non-integer), the program records an exception log and exits.

> [!NOTE]
>
> This environment variable is read when the `torch_npu._inductor` module is initialized.

PyTorch controls compilation concurrency through [TORCHINDUCTOR_COMPILE_THREADS](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py). In addition to the compilation concurrency configuration, TorchNPU provides `TORCHNPU_PRECOMPILE_THREADS` to independently control the number of precompile threads.

## Configuration Example

```bash
export TORCHNPU_PRECOMPILE_THREADS=16
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
