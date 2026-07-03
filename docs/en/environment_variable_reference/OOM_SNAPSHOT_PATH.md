# OOM\_SNAPSHOT\_PATH

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:04:58.145Z pushedAt=2026-06-16T03:14:22.298Z -->

## Feature Description

This environment variable configures the save path for memory snapshot data when an out-of-memory (OOM) error occurs.

- When not configured, the memory data is saved to the current path by default.
- When configured, memory snapshot data is saved to the specified path. Ensure that the path already exists and that the running process has write permission to it.

This environment variable is not configured by default.

## Configuration Example

```bash
export OOM_SNAPSHOT_PATH="/home/usr/"
```

## Usage Constraints

Must be used in conjunction with [OOM\_SNAPSHOT\_ENABLE](OOM_SNAPSHOT_ENABLE.md).

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
- <term>Atlas 800I A2 inference series</term>
- <term>Atlas inference series</term>
