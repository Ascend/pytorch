# ROOTINFO\_SCALABLE\_ENABLE

## Function Description

When HCCL is used as the communication backend and the communicator is established using the RootInfo method, this environment variable controls whether to enable Scalable RootInfo hierarchical link establishment.

- When set to "0" (default): Scalable RootInfo is disabled, and the communicator is established using the original single-RootInfo path.
- When set to "1": Scalable RootInfo is enabled. When the usage conditions are met, the ranks in the current communicator are continuously and evenly divided into multiple groups, and each group selects one root to establish the communicator in a hierarchical manner.

After this feature is enabled, the number of ranks that each root is expected to manage is configured by `TORCH_HCCL_RANKS_PER_ROOT`. When the number of ranks in the current communicator is not greater than the configured value, the original single-RootInfo path is still used.

This feature requires a CANN/HCCL version that supports the Scalable RootInfo interface. For details about how to use it, see [Scalable RootInfo Hierarchical Link Establishment](../../../developer_notes/distributed/communication_strategy/scalable_rootinfo_link_setup.md).

## Configuration Example

Enable Scalable RootInfo hierarchical link establishment:

```bash
export ROOTINFO_SCALABLE_ENABLE=1
```

Disable Scalable RootInfo hierarchical link establishment:

```bash
unset ROOTINFO_SCALABLE_ENABLE
```

## Usage Constraints

- This environment variable only affects the RootInfo link establishment path and does not change the rank table link establishment behavior.
- It takes effect only on HCCL communicators of the DEFAULT type. P2P communicators continue to use the original RootInfo path.
- HCCL communicators use a lazy loading mechanism. Link establishment is triggered only when a collective communication operation that requires a communicator is executed for the first time.
- This environment variable is cached after being read for the first time in a process. It should be configured before starting distributed workers. Dynamic switching within the same process is not supported.

## Supported Products

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>
<!-- end id2 -->
