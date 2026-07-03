# TORCH\_NPU\_LOGS\_FILTER

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:06:30.906Z pushedAt=2026-06-16T03:14:22.399Z -->

## Feature Description

This environment variable is used to filter the log output of Ascend Extension for PyTorch. It selects the log information to be displayed through a blocklist and allowlist mechanism, helping developers quickly locate key information among a large volume of logs.

When used in conjunction with `TORCH_NPU_LOGS`, it can further refine the control of log output content on the basis of enabling log printing, reducing interference from irrelevant logs and improving debugging efficiency.

Description of the blocklist and allowlist mechanism:

- Allowlist (+ prefix): Only displays log information that matches the specified keywords.
- Blocklist (- prefix): Filters out log information that matches the specified keywords.

This environment variable is not configured by default.

## Configuration Example

The following example uses the op_plugin module to demonstrate the filtering function, which is also applicable to other modules (memory, dispatch, etc.).

- Allowlist filtering:

    ```bash
    export TORCH_NPU_LOGS="+op_plugin"
    export TORCH_NPU_LOGS_FILTER="+aclnnAdd,+aclnnMul"
    ```

    The above configuration enables DEBUG level logs for the op_plugin module and only displays log information containing the "aclnnAdd" or "aclnnMul" operators.

- Blocklist filtering:

    ```bash
    export TORCH_NPU_LOGS="+op_plugin"
    export TORCH_NPU_LOGS_FILTER="-aclnnAdd,-aclnnMul"
    ```

    The above configuration enables DEBUG level logs for the op_plugin module and filters out log information containing the "aclnnAdd" or "aclnnMul" operators.

- Disable filtering:

    ```bash
    unset TORCH_NPU_LOGS_FILTER
    ```

## Usage Constraints

- Filtering rules are case-sensitive.
- Allowlist and denylist can be used together, and the denylist has a higher priority than the allowlist.
- Keyword matching uses substring matching, meaning that a log entry is considered a match if it contains the specified keyword.

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
- <term>Atlas 800I A2 inference series</term>
- <term>Atlas inference series</term>
