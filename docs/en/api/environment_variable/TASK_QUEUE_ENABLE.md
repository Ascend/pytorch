# TASK\_QUEUE\_ENABLE

## Feature Description

This environment variable configures whether the `task_queue` operator dispatch queue is enabled and its optimization level.

- When set to `0`: The `task_queue` operator dispatch queue optimization is disabled. The operator dispatch tasks are shown in Figure 1.

    **Figure 1**  Disabling task_queue
    ![figure1](../figures/stop_task_queue.png)

- When set to `1` or not configured: Level 1 optimization of the `task_queue` operator dispatch queue is enabled. The operator dispatch tasks are shown in Figure 2.

    Level 1 optimization: The `task_queue` operator dispatch queue optimization is enabled, dividing operator dispatch tasks into two stages. A portion of the tasks (mainly the invocation of `aclnn` operators) is placed on the newly added second-level pipeline. The first-level and second-level pipelines transfer tasks through the operator queue and run in parallel with each other, reducing the overall dispatch latency through partial masking and improving end-to-end performance.

    **Figure 2**  Level 1 optimization
    ![figure2](../figures/Level-1.png)

- When set to `2`: Level 2 optimization of the `task_queue` operator dispatch queue is enabled. The operator dispatch tasks are shown in Figure 3.

    Level 2 optimization: Includes the Level 1 optimization and further balances the task load between the first-level and second-level pipelines, mainly by migrating tasks related to `workspace` to the second-level pipeline, achieving better masking effects and greater performance gains. This configuration takes effect only in binary scenarios. You are advised to configure Level 2 optimization.

    **Figure 3**  Level 2 optimization
    ![figure3](../figures/Level-2.png)

    This environment variable is configured to `1` by default.

## Configuration Example

```bash
export TASK_QUEUE_ENABLE=2
```

## Usage Constraints

When [ASCEND\_LAUNCH\_BLOCKING](ASCEND_LAUNCH_BLOCKING.md) is set to `1`, the `task_queue` operator queue is disabled, and the `TASK_QUEUE_ENABLE` setting does not take effect.

When `TASK_QUEUE_ENABLE` is set to `2`, the peak NPU memory usage during runtime may increase due to memory concurrency.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas 800I A2 inference products</term>
- <term>Atlas inference products</term>
- <term>Ascend 950DT</term>
