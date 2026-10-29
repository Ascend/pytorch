# ACL\_DEVICE\_SYNC\_TIMEOUT

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

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT products</term>
<!-- end id5 -->
