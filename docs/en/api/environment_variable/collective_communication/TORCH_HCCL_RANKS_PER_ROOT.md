# TORCH\_HCCL\_RANKS\_PER\_ROOT

## Function Description

After Scalable RootInfo hierarchical link establishment is enabled, this environment variable configures the maximum number of ranks that each root is expected to manage, and calculates the number of roots used by the current communicator accordingly.

The default value of this environment variable is "128", and the value range is decimal integers from 1 to 4294967295.

Assume that the current ProcessGroup contains N ranks, and the value of this environment variable is R:

- When N is less than or equal to R: The original single-RootInfo path is used.
- When N is greater than R: The number of roots K is `ceil(N / R)`. The N ranks are continuously and evenly divided into K groups, with at most a difference of 1 in the number of ranks between groups. The first `local rank` of each group serves as the root.

## Configuration Example

Enable Scalable RootInfo and set the number of ranks that each root is expected to manage to 128:

```bash
export ROOTINFO_SCALABLE_ENABLE=1
export TORCH_HCCL_RANKS_PER_ROOT=128
```

When only the feature switch is configured, this environment variable uses the default value "128":

```bash
export ROOTINFO_SCALABLE_ENABLE=1
unset TORCH_HCCL_RANKS_PER_ROOT
```

## Usage Constraints

- It participates in communicator path selection only when `ROOTINFO_SCALABLE_ENABLE=1` and the device supports Scalable RootInfo.
- If the value is an empty string, 0, a negative number, a non-numeric value, a value containing extra characters, or a value exceeding 4294967295, a parameter error is returned.
- The environment variable should be configured before starting distributed workers. For details about how to use it, see [Scalable RootInfo Hierarchical Link Establishment](../../../developer_notes/distributed/communication_strategy/scalable_rootinfo_link_setup.md).

## Supported Products

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>
<!-- end id2 -->
