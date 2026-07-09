# ACL\_DEVICE\_SYNC\_TIMEOUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:02:44.860Z pushedAt=2026-06-16T03:14:22.111Z -->

## Function Description

This environment variable configures the timeout for device synchronization.

The unit is second (s), and the configuration range is [1, 2147483]. If not configured, the default device synchronization timeout is used.

## Configuration Example

```shell
export ACL_DEVICE_SYNC_TIMEOUT=300
```

## Usage Constraints

None

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
