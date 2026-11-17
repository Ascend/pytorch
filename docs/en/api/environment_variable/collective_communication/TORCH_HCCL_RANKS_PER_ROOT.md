# TORCH\_HCCL\_RANKS\_PER\_ROOT

## Function Description

This environment variable controls Scalable RootInfo hierarchical link establishment and configures the maximum number of ranks that each root is expected to manage.

The default value of this environment variable is "-1", indicating that Scalable RootInfo is disabled and the original RootInfo link establishment path is used. When set to a decimal integer from 1 to 4294967295, it indicates the maximum number of ranks that each root is expected to manage.

When set to a positive integer, assume that the current ProcessGroup contains N ranks, and the value of this environment variable is R:

- When N is less than or equal to R: The original single-RootInfo path is used.
- When N is greater than R: The number of roots K is `ceil(N / R)`. The N ranks are continuously and evenly divided into K groups, with at most a difference of 1 in the number of ranks between groups. The first `local rank` of each group serves as the root.

## Configuration Example

Enable Scalable RootInfo hierarchical link establishment:

```bash
export TORCH_HCCL_RANKS_PER_ROOT=128
```

Disable Scalable RootInfo:

```bash
export TORCH_HCCL_RANKS_PER_ROOT=-1
```

If this environment variable is not set or is unset with `unset TORCH_HCCL_RANKS_PER_ROOT`, the default value "-1" is used, and the feature is disabled.

## Usage Constraints

- When set to a positive integer and the device supports Scalable RootInfo, it participates in communicator path selection.
- If the value is an empty string, 0, a negative number less than -1, a non-numeric value, a value containing extra characters, or a value exceeding 4294967295, a parameter error is returned.
- The environment variable should be configured before starting distributed workers. For details about how to use it, see [Scalable RootInfo Hierarchical Link Establishment](../../../developer_notes/distributed/communication_strategy/scalable_rootinfo_link_setup.md).

## Supported Products

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>
<!-- end id2 -->
