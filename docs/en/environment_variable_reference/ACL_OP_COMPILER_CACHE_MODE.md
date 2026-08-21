# ACL\_OP\_COMPILER\_CACHE\_MODE

## Function Description

This environment variable configures the disk cache mode for operator compilation.

- Configured as `enable`: enables the operator compilation cache. Once enabled, it avoids repeated compilation of operators with the same compilation parameters and operator parameters, thereby improving compilation speed.
- Configured as `disable`: disables the operator compilation cache.
- Configured as `force`: forcibly refreshes the cache. When your Python or dependency libraries change, you need to set the mode to `force` to clear the existing cache.

The default configuration is `enable`.

## Configuration Example

```bash
export ACL_OP_COMPILER_CACHE_MODE=enable
```

## Usage Constraints

- This environment variable can only be used in single-operator mode. Graph mode does not support this environment variable.
- When operator compilation cache is enabled, you can configure the storage path of operator compilation cache files through [ACL\_OP\_COMPILER\_CACHE\_DIR](ACL_OP_COMPILER_CACHE_DIR.md).
- If the operator compilation disk cache mode is specified through both this environment variable and the torch_npu_option method, the torch_npu_option method in the code takes precedence.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas 800I A2 inference products</term>
