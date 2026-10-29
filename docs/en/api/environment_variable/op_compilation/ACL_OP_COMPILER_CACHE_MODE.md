# ACL\_OP\_COMPILER\_CACHE\_MODE

## Function Description

This environment variable configures the disk cache mode for operator compilation.

- Configured as `enable`: enables the operator compilation cache. Once enabled, it avoids repeated compilation of operators with the same compilation parameters and operator parameters, thereby improving compilation speed.
- Configured as `disable`: disables the operator compilation cache.
- Configured as `force`: forcibly refreshes the cache. When your Python or dependency libraries change, you need to set the mode to `force` to clear the existing cache.

The default configuration is `enable`.

This environment variable is provided by TorchNPU. PyTorch does not have a directly corresponding environment variable.

## Configuration Example

```bash
export ACL_OP_COMPILER_CACHE_MODE=enable
```

## Usage Constraints

- This environment variable can only be used in single-operator mode and is not supported in graph mode.
- When operator compilation cache is enabled, you can configure the storage path of operator compilation cache files through [ACL\_OP\_COMPILER\_CACHE\_DIR](ACL_OP_COMPILER_CACHE_DIR.md).
- If the operator compilation disk cache mode is specified through both this environment variable and the torch_npu_option method, the torch_npu_option method in the code takes precedence.

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
<!-- npu="910b" id4 -->
- <term>Atlas 800I A2</term>
<!-- end id4 -->
