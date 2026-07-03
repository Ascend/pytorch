# ACL\_OP\_COMPILER\_CACHE\_MODE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:02:47.859Z pushedAt=2026-06-16T03:14:22.124Z -->

## Feature Description

This environment variable configures the disk cache mode for operator compilation.

- Configured as `enable`: Enables the operator compilation cache. Once enabled, it avoids repeated compilation of operators with the same compilation parameters and operator parameters, thereby improving compilation speed.
- Configured as `disable`: Disables the operator compilation cache.
- Configured as `force`: Force refresh the cache. When the user's Python or dependency libraries change, it needs to be set to `force` to clear the existing cache.

The default configuration is `enable`.

## Configuration Example

```bash
export ACL_OP_COMPILER_CACHE_MODE=enable
```

## Usage Constraints

- This environment variable can only be used in single-operator mode. Graph mode does not support this environment variable.
- When operator compilation cache is enabled, the storage path for operator compilation cache files can be configured via [ACL_OP_COMPILER_CACHE_DIR](ACL_OP_COMPILER_CACHE_DIR.md).
- If the operator compilation disk cache mode is specified through both this environment variable and the torch_npu_option method, the torch_npu_option method in the code takes precedence.

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
- <term>Atlas 800I A2 inference series</term>
