# ACL\_OP\_COMPILER\_CACHE\_DIR

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:02:45.850Z pushedAt=2026-06-16T03:14:22.118Z -->

## Function Description

This environment variable configures the directory for the operator compilation disk cache.

Priority: `ACL_OP_COMPILER_CACHE_DIR` > `ASCEND_CACHE_PATH` > default path (`$HOME/atc_data`)

If this environment variable is set, the operator compilation cache is written to the disk at the specified path of this environment variable; if not set, the path specified by `ASCEND_CACHE_PATH` is used; if this environment variable is not set and no path is specified by `ASCEND_CACHE_PATH`, the default path (`$HOME/atc_data`) is used.

## Configuration Example

```bash
export ACL_OP_COMPILER_CACHE_DIR=/home/cache
```

## Usage Constraints

- This environment variable can only be used in single-operator mode and is not supported in graph mode.
- This environment variable must be used in conjunction with `ACL_OP_COMPILER_CACHE_MODE`.
- If both the environment variable and torch_npu_option are set, the torch_npu_option method in the code takes precedence.
- If the `ACL_OP_DEBUG_LEVEL` compilation option is set, the compilation cache feature is enabled only when the compilation option value is 0 or 3; other values disable the compilation cache feature. For details about the ACL_OP_DEBUG_LEVEL compilation option, see the "[aclCompileOpt](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendgraphapi/aclcppdevg_03_1371.html)" section in the *CANN GE APIs*.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas 800I A2 inference products</term>
