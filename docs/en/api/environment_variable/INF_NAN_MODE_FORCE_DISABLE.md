# INF\_NAN\_MODE\_FORCE\_DISABLE

## Function Description

For Atlas A2 training series/Atlas A3 training series, this environment variable can be used to force disable INF_NAN mode. Disabling INF_NAN mode causes Inf and NaN generated during computation to be converted to the maximum value and 0 of the corresponding data type, leading to differences in subsequent operation results. Therefore, a forced interception is implemented. If you must disable INF_NAN mode, set this environment variable to "1" to forcibly disable INF_NAN mode. After disabling it, pay attention to the changes in Inf and NaN values.

- 1: Force disable INF_NAN mode and enable saturation mode. After forcibly disabling INF_NAN mode on Atlas A2 training series/Atlas A3 training series, pay attention to the changes in Inf and NaN values.
- 0: Does not process INF_NAN mode. When INF_NAN mode is disabled on Atlas A2 training series/Atlas A3 training series, an error is intercepted and reported. The default value is 0.

For Atlas training series/Atlas inference series/Atlas 200I/500 A2 inference products/Ascend 950DT, this environment variable does not take effect.

## Configuration Example

```bash
export INF_NAN_MODE_FORCE_DISABLE=1
```

## Usage Constraints

None

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
