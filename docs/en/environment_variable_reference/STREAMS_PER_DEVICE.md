# STREAMS\_PER\_DEVICE

## Feature Description

This environment variable configures the maximum number of streams in the stream pool.

The stream pool uses a Round Robin strategy.

- When configured as 32: The stream pool has 32 streams.
- When configured as 8: The stream pool has 8 streams.
- When configured as any other value: a Warning-level log is printed, and the environment variable is set to the default value 32.

The default value of this environment variable is 32.

## Configuration Example

```bash
export STREAMS_PER_DEVICE=8
```

## Usage Constraints

None

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950DT</term>
