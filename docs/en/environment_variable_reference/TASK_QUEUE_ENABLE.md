# TASK\_QUEUE\_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:06:09.685Z pushedAt=2026-06-16T03:14:22.383Z -->

## Feature Description

This environment variable configures whether to enable the task_queue operator dispatch queue and its optimization level.

- When configured as "0": Disables the task_queue operator dispatch queue optimization. The operator dispatch task is shown in [Figure 1](#disabling-task_queue).

    **Figure 1** Disabling task_queue<a id="disabling-task_queue"></a>    
    ![figure 1](../figures/stop_task_queue.png)

- When configured as "1" or not configured: Enables Level 1 optimization of the `task_queue` operator dispatch queue. The operator dispatch tasks are shown in [Figure 2](#Level-1-optimization).

    Level 1 optimization: Enables the task_queue operator dispatch queue optimization, splitting operator dispatch tasks into two segments. Some tasks (primarily aclnn operator calls) are placed on the newly added secondary pipeline. The primary and secondary pipelines transfer tasks through the operator queue and run in parallel, partially overlapping to reduce the overall dispatch time and improve end-to-end performance.

    **Figure 2** Level 1 optimization<a id="Level-1-optimization"></a>     
    ![figure 2](../figures/Level-1.png)

- When configured as "2": Enables Level 2 optimization of the `task_queue` operator dispatch queue. The operator dispatch tasks are shown in [Figure 3](#Level-2-optimization).

    Level 2 optimization: Includes Level 1 optimization and further balances the task load between the primary and secondary pipelines, primarily by migrating workspace-related tasks to the secondary pipeline. This achieves better overlap and greater performance gains. This configuration takes effect only in binary scenarios and is recommended to be configured as Level 2 optimization.

    **Figure 3**  Level 2 optimization<a id="Level-2-optimization"></a>       
    ![figure 3](../figures/Level-2.png)

    This environment variable is configured as "1" by default.

## Configuration example

```bash
export TASK_QUEUE_ENABLE=2
```

## Usage Constraints

When [ASCEND\_LAUNCH\_BLOCKING](ASCEND_LAUNCH_BLOCKING.md) is set to "1", the task\_queue operator queue is disabled, and the TASK\_QUEUE\_ENABLE setting does not take effect.

When `TASK_QUEUE_ENABLE` is configured as "2", the NPU memory peak may increase during runtime due to memory concurrency.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas 800I A2 inference products</term>
- <term>Atlas inference products</term>
