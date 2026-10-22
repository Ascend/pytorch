# OOM\_SNAPSHOT\_ENABLE

## Function Description

This environment variable configures whether to save memory data when an out-of-memory (OOM) error occurs, for analysis of the cause of the memory shortage.

- When set to "2", only the current memory usage information, including allocated and freed memory information, is saved when an out-of-memory (OOM) error occurs.
- When set to "1", both the current and historical memory usage information, including allocated and freed memory information, is saved when an out-of-memory (OOM) error occurs.
- When set to "0", this function is disabled and memory data is not saved.

This environment variable defaults to "0".

## Configuration Example

```bash
export OOM_SNAPSHOT_ENABLE=1
```

## Usage Constraints

None

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
