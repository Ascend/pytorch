# OOM\_SNAPSHOT\_ENABLE

## Function Description

This environment variable configures whether to save memory data when an out-of-memory (OOM) error occurs, for analysis of the cause of the memory shortage.

- When set to "2", only the current memory usage information, including allocated and freed memory information, is saved when an out-of-memory (OOM) error occurs.
- When set to "1", both the current and historical memory usage information, including allocated and freed memory information, is saved when an out-of-memory (OOM) error occurs.
- When set to "0", this function is disabled and memory data is not saved.

This environment variable defaults to 0.

## Configuration Example

```bash
export OOM_SNAPSHOT_ENABLE=1
```

## Usage Constraints

None

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas 800I A2 inference products</term>
- <term>Atlas inference products</term>
- <term>Ascend 950DT</term>
