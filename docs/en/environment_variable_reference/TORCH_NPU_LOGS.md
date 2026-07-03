# TORCH\_NPU\_LOGS

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:06:39.714Z pushedAt=2026-06-16T03:14:22.401Z -->

## Function Description

This environment variable is used to configure the log printing function of new modules in Ascend Extension for PyTorch, providing developers with precise debugging and locating capabilities in debugging scenarios.

The new modules added by Ascend Extension for PyTorch do not support configuration through the native `TORCH_LOGS`. To set the log information for the new modules, you need to use `TORCH_NPU_LOGS`. The list of new modules is as follows:

| Field Name | Corresponding Module | Function Description |
| --- | ---- | --- |
| memory | Memory Management | Print memory-related logs |
| dispatch | Operator Dispatch | Print operator dispatch-related logs |
| dispatch\_time | Operator Dispatch Time | Print operator dispatch time logs |
| silent | Silent Detection | Print silent detection-related logs |
| recovery | Process-Level Online Recovery | Print process-level online recovery-related logs |
| op\_plugin | Operator Adaptation | Print operator adaptation-related logs |
| shmem | Shared Memory | Print shared memory-related logs |
| env | Environment Variable | Print environment variable call logs |
| acl | acl | Print acl-related logs |
| aclgraph | aclgraph | Print aclgraph-related logs |

Ascend Extension for PyTorch enhances the native logging functionality, supporting log printing on the C++ side.

- When configured, logging information printing is enabled, and the log information of the specified modules will be printed normally on the screen of the primary node.

- When not configured, logging information printing is disabled, and log information will not be printed on the screen.

This environment variable is not configured by default.

## Configuration Example

- Enable logging information output:

    ```bash
    export TORCH_NPU_LOGS=memory,+dispatch,-all
    ```

    Level description:

    - : INFO level, the default level, outputs general runtime information.
    - +: DEBUG level, outputs the most detailed debugging information.
    - -: ERROR level, outputs only error and warning information.

    The above configuration example indicates that memory (memory management) prints INFO-level logs, dispatch (operator dispatch) prints DEBUG-level logs, and all (all other modules, including native PyTorch and new modules added by Ascend Extension for PyTorch) prints ERROR-level log information.

- Disable logging log information printing:

    ```bash
    unset TORCH_NPU_LOGS
    ```

## Usage Constraints

The shmem module takes effect only on PyTorch 2.7.1 and later versions.

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
- <term>Atlas 800I A2 inference series</term>
- <term>Atlas inference series</term>
