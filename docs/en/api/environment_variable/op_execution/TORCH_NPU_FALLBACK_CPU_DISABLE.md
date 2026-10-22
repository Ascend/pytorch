# TORCH\_NPU\_FALLBACK\_CPU\_DISABLE

## Function Description

This environment variable controls whether the implicit CPU fallback path managed by TorchNPU is allowed to execute. When there is no corresponding operator implementation on the NPU side, or the managed compatibility path of an operator needs to complete the main numerical computation on the CPU, you can use this environment variable to choose to maintain compatible execution or report an error directly.

- When configured as "0", CPU fallback is allowed. When this environment variable is not configured, it is treated as "0" by default to maintain the existing compatible behavior. The CPU fallback path can copy the input from the NPU to the CPU to complete the computation and then copy the result back to the NPU. Paths with alarms configured print a fallback alarm.
- When configured as "1", CPU fallback is prohibited. The operator reports an error directly before entering the main CPU computation. The error message contains the operator name and `TORCH_NPU_FALLBACK_CPU_DISABLE=1`, and the "About to fall back to CPU" alarm is not printed.

This environment variable is provided by TorchNPU. PyTorch does not have a directly corresponding environment variable.

## Configuration Example

Prohibiting CPU fallback of managed operators:

```bash
export TORCH_NPU_FALLBACK_CPU_DISABLE=1
```

Allowing CPU fallback of managed operators:

```bash
export TORCH_NPU_FALLBACK_CPU_DISABLE=0
```

## Usage Constraints

- This environment variable supports only "0" or "1". The default value is "0".
- This environment variable must be configured before the Python process starts. TorchNPU caches the configuration value after the first read in the process; modifying `os.environ` during runtime does not change the behavior of the current process. To switch the configuration, start a new Python process.
- This environment variable only controls the gated CPU fallback paths of TorchNPU, including the Dispatcher fallback hit when no NPU kernel is registered, and the gated explicit CPU kernels. Scenarios such as the ACL Op/aclops path, AclNN compatibility path, control-type CPU computation, Host staging, Host weight preprocessing, explicit user calls to `.cpu()`, or explicit creation of CPU Tensors are not within the control scope of this environment variable.
- This environment variable does not change operator registration relationships or add NPU implementations for operators. When configured as "1", operators that lack an NPU implementation and hit the managed CPU fallback path report an error directly, which may affect the compatibility of existing models. It is recommended to use it in operator supportability checks, performance troubleshooting, or scenarios requiring strict NPU execution.

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
<!-- npu="910b" id6 -->
- <term>Atlas 800I A2</term>
<!-- end id6 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT products</term>
<!-- end id5 -->
