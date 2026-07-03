# OOM\_SNAPSHOT\_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:04:54.815Z pushedAt=2026-06-16T03:14:22.292Z -->

## Feature Description

This environment variable configures whether to save memory data when an Out-of-Memory (OOM) Report occurs, for analyzing the cause of the memory shortage.

- When set to "2", only the current memory usage information, including allocated and freed memory information, is saved when an Out-of-Memory (OOM) Report occurs.
- When set to "1", both the current and historical memory usage information, including allocated and freed memory information, is saved when an Out-of-Memory (OOM) Report occurs.
- Configured as "0", this function is disabled and memory data is not saved.

This environment variable defaults to 0.

## Configuration Example

```bash
export OOM_SNAPSHOT_ENABLE=1
```

## Usage Constraints

None

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
- <term>Atlas 800I A2 inference series</term>
- <term>Atlas inference series</term>
