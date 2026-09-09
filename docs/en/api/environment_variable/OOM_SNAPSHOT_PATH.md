# OOM\_SNAPSHOT\_PATH

## Function Description

This environment variable configures the save path of memory data when an out-of-memory (OOM) error occurs.

- When not configured, the memory data is saved to the current path by default.
- When configured, the memory data is saved to the specified path. Ensure that the path already exists and that the running process has write permission to it.

This environment variable is not configured by default.

## Configuration Example

```bash
export OOM_SNAPSHOT_PATH="/home/usr/"
```

## Usage Constraints

Must be used in conjunction with [OOM\_SNAPSHOT\_ENABLE](OOM_SNAPSHOT_ENABLE.md).

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas 800I A2 inference products</term>
- <term>Atlas inference products</term>
- <term>Ascend 950DT</term>
