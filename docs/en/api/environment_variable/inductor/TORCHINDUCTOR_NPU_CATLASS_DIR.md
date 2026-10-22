# TORCHINDUCTOR\_NPU\_CATLASS\_DIR

## Function Description

This environment variable configures the path of the Catlass template library. Catlass is a matrix multiplication template acceleration library on the NPU platform. During autotune, Inductor can use the Catlass template library to perform template tuning on matmul operators.

- Default value not configured: uses the `../third_party/catlass` path under the `torch` installation directory (source compilation scenario); if the path does not exist, the Catlass template library is not loaded.
- Set to a valid path: loads the Catlass template library from the configured path.
- Set to an invalid path: loading the Catlass template library fails, a WARNING log is printed, and the Catlass template library is automatically skipped, without affecting the tuning of other candidate implementations.

> [!NOTE]
>
> - This environment variable must be set before importing `torch_npu`.
> - Ensure that the path is valid and the Catlass template library is correctly installed.
> - Using the Catlass template library requires configuring `TORCHINDUCTOR_MAX_AUTOTUNE=1` and including `"CATLASS"` in `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`.

This variable corresponds to PyTorch's [TORCHINDUCTOR_CUTLASS_DIR](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py). PyTorch configures the CUTLASS (CUDA Templates for Linear Algebra Subroutines) library path, while TorchNPU configures the Catlass library path.

## Configuration Example

```bash
export TORCHINDUCTOR_NPU_CATLASS_DIR="/path/to/catlass/dir"
```

## Usage Constraints

None

## Supported Products

<!-- npu="950" id1 -->
<term>Ascend 950DT products</term>
<!-- end id1 -->
