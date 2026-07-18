# KINETO\_USE\_DAEMON

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:04:11.133Z pushedAt=2026-06-16T03:14:22.236Z -->

## Feature Description

This environment variable is used to set whether to enable dynamic\_profile collection through msMonitor nputrace in training scenarios.

## Configuration Example

```bash
export KINETO_USE_DAEMON=1
```

For detailed usage, see the "[dynamic\_profile](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0033.html)" section in the *CANN Performance Tuning Tool*.

## Usage Constraints

- When no code is manually added to the script, this environment variable applies to PyTorch training scenarios.

- After the dynamic_profile module is added to the script, this environment variable can be used in non-training scenarios. For example:

    ```python
    # Load the dynamic_profile module
    from torch_npu.profiler import dynamic_profile as dp
    # Set the path to the profiling configuration file
    dp.init("profiler_config_path")
    …
    for step in steps:
        train_one_step()
        # Mark the step
        dp.step()
    ```

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
  