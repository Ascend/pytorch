# PER\_STREAM\_QUEUE

> [!NOTICE]  
> This feature is currently experimental and may change in future releases.

## Function Description

Whether to enable the one task\_queue operator dispatch queue per stream can be configured through this environment variable.

- When configured as "0", the one task\_queue operator dispatch queue per stream is disabled.
- When configured as "1", the one task\_queue operator dispatch queue per stream is enabled.

This environment variable is configured as "0" by default.

## Configuration Example

```bash
export PER_STREAM_QUEUE=1
```

## Usage Constraints

This environment variable takes effect only when [TASK_QUEUE_ENABLE](TASK_QUEUE_ENABLE.md) is configured as "1" or "2".

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
- <term>Atlas 800I A2</term>
<!-- end id6 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT products</term>
<!-- end id5 -->
