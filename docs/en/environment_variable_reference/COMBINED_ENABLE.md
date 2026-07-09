# COMBINED\_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:03:09.159Z pushedAt=2026-06-16T03:14:22.150Z -->

## Function Description

This environment variable sets the combined flag. Setting it to 0 disables this feature; setting it to 1 enables it, which is used to optimize scenarios involving two non-consecutive operator combination classes.

The default configuration is 0.

## Configuration Example

```bash
export COMBINED_ENABLE=1
```

## Usage Constraints

None

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
