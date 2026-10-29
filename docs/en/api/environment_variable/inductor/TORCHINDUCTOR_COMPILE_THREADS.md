# TORCHINDUCTOR\_COMPILE\_THREADS

## Function Description

This environment variable configures the number of processes for concurrent compilation. During compilation, Inductor uses multiple processes to compile kernels in parallel, speeding up the compilation process.

- Default value `min(32, number of CPU cores)`: the number of concurrent compilation processes is the smaller of 32 and the number of CPU cores on the machine.
- Set to "1": compiles serially in the main process without enabling compilation subprocesses; this can be used to locate compilation problems.
- Set to a positive integer greater than 1: uses the specified number of processes for concurrent compilation.

> [!NOTE]
>
> - This environment variable must be set before importing `torch_npu`.
> - Increasing the number of threads can speed up compilation but increases memory usage.
> - This variable has no built-in upper limit. An excessively large value spawns too many compilation subprocesses at the first compilation, which may exhaust memory or cause process creation failures. It is recommended not to exceed the number of CPU cores on the machine.
> - In Ascend C mode, the upper limit of concurrent compilation processes is 32.
> - On NPU, this environment variable is also used as the derivation input for the default value of `TORCHNPU_PRECOMPILE_THREADS` (`number of CPU cores // max(that value, 2)`, and not less than 32).

This variable corresponds to PyTorch's [TORCHINDUCTOR_COMPILE_THREADS](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py), with the same configuration method and a default value of `min(32, number of CPU cores)`.

## Configuration Example

```bash
export TORCHINDUCTOR_COMPILE_THREADS=16
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
