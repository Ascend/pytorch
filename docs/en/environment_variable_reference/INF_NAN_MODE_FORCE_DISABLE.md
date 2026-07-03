# INF\_NAN\_MODE\_FORCE\_DISABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:04:11.615Z pushedAt=2026-06-16T03:14:22.242Z -->

## Feature Description

For <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>, this environment variable can be used to force disable INF_NAN mode. Disabling INF_NAN mode causes Inf and NaN generated during computation to be converted to the maximum value and 0 of the corresponding data type, leading to discrepancies in subsequent operation results, so a forced interception is implemented. If you must disable INF_NAN mode, you need to set this environment variable to "1", which force disables INF_NAN mode. After disabling, pay attention to the changes in Inf and NaN values.

- 1: Force disable INF_NAN mode and enable saturation mode. After force disabling INF_NAN mode on <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>, pay attention to the changes in Inf and NaN values.

- 0: Do not process INF_NAN mode. When INF_NAN mode is disabled on <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>, an error will be intercepted and reported. The default value is 0.

For <term>Atlas training products</term>/<term>Atlas inference products</term>/<term>Atlas 200I/500 A2 inference products</term>/<term>Atlas 350 accelerator card</term>, this environment variable does not take effect.

## Configuration example

```bash
export INF_NAN_MODE_FORCE_DISABLE=1
```

## Usage Constraints

None

## Supported Products

- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
