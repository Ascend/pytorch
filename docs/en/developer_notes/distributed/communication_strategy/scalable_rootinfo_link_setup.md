# Scalable RootInfo Hierarchical Link Setup

## Introduction

As the distributed cluster scale grows, the number of ranks in an HCCL communicator also increases. In the traditional RootInfo link setup method, a single root handles the information negotiation required for communicator initialization. The processing pressure on a single root increases with the communicator scale, which may become a link setup performance bottleneck for large-scale clusters.

Scalable RootInfo uses a hierarchical link setup method that divides the ranks in a communicator into multiple groups consecutively and evenly, with multiple roots jointly completing information negotiation. This reduces the number of ranks handled by a single root and improves the link setup efficiency of large-scale communicators.

The Scalable RootInfo feature is disabled by default. When disabled, the original RootInfo link setup behavior remains unchanged.

## Application Scenarios

When single-root negotiation becomes the performance bottleneck of communicator initialization in a large-scale cluster, you can use Scalable RootInfo hierarchical link setup to reduce the number of ranks handled by a single root.

This feature applies to the default global ProcessGroup and non-global ProcessGroups created via `torch.distributed.new_group`. The group size is calculated based on the number of ranks in the current ProcessGroup rather than the global world size.

## Usage Guide

Enable Scalable RootInfo through environment variables and configure the number of ranks that each root is expected to manage:

- `ROOTINFO_SCALABLE_ENABLE`: The switch of the Scalable RootInfo feature. The default value is "0". Set it to "1" to enable the feature.
- `TORCH_HCCL_RANKS_PER_ROOT`: The maximum number of ranks that each root is expected to manage. The default value is "128", and the value must be a decimal integer from 1 to 4294967295.

For the complete description of the environment variables, see the "[ROOTINFO\_SCALABLE\_ENABLE](../../../api/environment_variable/collective_communication/ROOTINFO_SCALABLE_ENABLE.md)" and "[TORCH\_HCCL\_RANKS\_PER\_ROOT](../../../api/environment_variable/collective_communication/TORCH_HCCL_RANKS_PER_ROOT.md)" sections in Environment Variables.

Assume the current ProcessGroup contains N ranks and `TORCH_HCCL_RANKS_PER_ROOT` is R:

- When N is not greater than R, the original single-RootInfo path continues to be used.
- When N is greater than R, the number of roots K is `ceil(N / R)`, and the Scalable RootInfo path is entered.
- The N ranks are divided into K groups consecutively and evenly. The number of ranks in each group differs by at most 1, and the first local rank of each group acts as the root.

## Usage Examples

Enable Scalable RootInfo and set the number of ranks that each root is expected to manage to 128:

```bash
export ROOTINFO_SCALABLE_ENABLE=1
export TORCH_HCCL_RANKS_PER_ROOT=128
```

For example, when a communicator contains 8 ranks and R is 2, K is 4, the root local ranks are 0, 2, 4, and 6 respectively, and each root manages 2 ranks. When a communicator contains 8 ranks and R is 3, K is 3, the three groups contain 3, 3, and 2 ranks respectively, and the root local ranks are 0, 3, and 6 respectively.

To disable Scalable RootInfo, run:

```bash
unset ROOTINFO_SCALABLE_ENABLE
unset TORCH_HCCL_RANKS_PER_ROOT
```

## Constraints

<!-- npu="A3,910b" id1 -->
- Only <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term> support Scalable RootInfo. For other models, even if the feature is enabled, a Warning is printed and the original RootInfo path is used instead.
<!-- end id1 -->
- A CANN/HCCL version containing the `HcclGetRootInfoScalable` and `HcclCommInitRootInfoScalable` interfaces is required. If the Scalable path has been selected on supported devices but the runtime library lacks these interfaces, a not-supported error is returned.
- This feature only affects the RootInfo link setup path and does not change the ranktable link setup behavior. When a valid `RANK_TABLE_FILE` is configured and the ranktable path is selected, the Scalable RootInfo path is not entered.
- P2P communicators continue to use the original RootInfo path.
- The environment variables must be configured before starting distributed workers. Dynamically switching the feature switch within the same process is not supported.
- Scalable RootInfo only changes the communicator initialization method, not the way communication operators are called or the way communicators are cached and destroyed.
