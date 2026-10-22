# ENABLE\_PARALLEL\_SCHEDULER

## Function Description

This environment variable controls whether to enable the parallel scheduler FX pass. When enabled, `torch_npu` groups and reorders scheduling nodes by operator type (Cube/Vector) during graph optimization, splitting matrix multiplication (Cube) and vector (Vector) operators into different pipeline groups, improving parallelism between computation and communication and between large and small operators.

- Default value `false`: the parallel scheduler is not enabled.
- Set to `true`: enables the parallel scheduler.
- Set to other values such as "1": the parallel scheduler is not enabled; only `true` (case-insensitive) takes effect.

> [!NOTE]
>
> - This environment variable must be set before importing `torch_npu`.
> - When the total number of nodes involved in scheduling does not exceed the built-in threshold (20 by default), parallel scheduling is not performed and the original scheduling is used.

This environment variable is provided by TorchNPU; PyTorch has no directly corresponding environment variable.

## Configuration Example

```bash
export ENABLE_PARALLEL_SCHEDULER=true
```

## Usage Constraints

It only takes effect for NPU wrapper code generation (NPUPythonWrapperCodeGen).

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="950" id2 -->
- <term>Ascend 950DT products</term>
<!-- end id2 -->
