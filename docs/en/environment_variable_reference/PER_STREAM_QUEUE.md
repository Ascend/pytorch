# PER\_STREAM\_QUEUE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:05:15.594Z pushedAt=2026-06-16T03:14:22.315Z -->

> [!NOTICE]  
> This feature is currently experimental and may change in future releases.

## Feature Description

Whether to enable the one task\_queue operator dispatch queue per stream can be configured through this environment variable.

- When configured as "0", the one task\_queue operator dispatch queue per stream is disabled.

- When configured as "1", the one task\_queue operator dispatch queue per stream is enabled.

This environment variable is configured as "0" by default.

## Configuration example

```bash
export PER_STREAM_QUEUE=1
```

## Usage Constraints

This environment variable takes effect only when [TASK_QUEUE_ENABLE](TASK_QUEUE_ENABLE.md) is configured as "1" or "2".

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas 800I A2 inference products</term>
- <term>Atlas inference products</term>
