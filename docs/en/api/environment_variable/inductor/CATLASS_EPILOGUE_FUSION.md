# CATLASS\_EPILOGUE\_FUSION

## Function Description

This environment variable controls whether to enable the Catlass epilogue fusion feature. When enabled, the Catlass template library fuses epilogue operations (such as bias add, activation, etc.) into the matmul kernel, reducing additional kernel launches and memory copies.

- Default value "0": disables Catlass epilogue fusion.
- Set to "1": enables Catlass epilogue fusion.

> [!NOTE]
>
> - This environment variable must be set before importing `torch_npu`.
> - Enabling Catlass epilogue fusion also requires configuring `TORCHINDUCTOR_NPU_CATLASS_DIR` (specifying the Catlass template library path) and `TORCHINDUCTOR_MAX_AUTOTUNE=1`.

This variable corresponds to PyTorch's [CUTLASS_EPILOGUE_FUSION](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py), with the same configuration method and a default value of "0".

## Configuration Example

Enabling Catlass epilogue fusion:

```bash
export CATLASS_EPILOGUE_FUSION=1
```

## Usage Constraints

None

## Supported Products

<!-- npu="950" id1 -->
<term>Ascend 950DT products</term>
<!-- end id1 -->
