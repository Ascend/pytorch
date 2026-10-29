# TORCH\_NPU\_LOGS

## Function Description

This environment variable is used to configure the log printing function of new TorchNPU modules, providing you with precise debugging and locating capabilities in debugging scenarios.

The new TorchNPU modules do not support configuration through the native `TORCH_LOGS`. To set the log information of the new modules, you need to use `TORCH_NPU_LOGS`. The list of new modules is as follows:

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
| acl | acl | Print ACL-related logs |
| aclgraph | aclgraph | Print aclgraph-related logs |
| fxrt | FXRT Graph Runtime | Print FXRT-related logs. The log content carries the `[FXRT]` prefix |

TorchNPU enhances the native logging print function and supports log printing on the C++ side.

- When configured, logging information printing is enabled, and the log information of the specified modules is printed on the screen of the primary node.
- When not configured, logging information printing is disabled, and log information is not printed on the screen.

This environment variable is not configured by default.

## Configuration Example

- Enable logging information printing:

    ```bash
    export TORCH_NPU_LOGS=memory,+dispatch,-all
    ```

    Level description:

    - No sign (default): INFO level, the default level, outputs general runtime information.

    - +: DEBUG level, outputs the most detailed debugging information.

    - -: ERROR level, outputs only error and warning information.

    The preceding configuration example indicates that memory (memory management) prints INFO-level logs, dispatch (operator dispatch) prints DEBUG-level logs, and all (all other modules, including native modules and new modules added by TorchNPU) prints ERROR-level log information.

- Disable logging information printing:

    ```bash
    unset TORCH_NPU_LOGS
    ```

## Usage Constraints

The shmem module takes effect only on PyTorch 2.7.1 and later versions.

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
- <term>Atlas 800I A2 inference server</term>
<!-- end id6 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT products</term>
<!-- end id5 -->
