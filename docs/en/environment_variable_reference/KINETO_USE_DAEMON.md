# KINETO\_USE\_DAEMON

## Feature Description

This environment variable is used to set whether to enable dynamic\_profile collection through msMonitor nputrace in training scenarios.

## Configuration Example

```bash
export KINETO_USE_DAEMON=1
```

For detailed usage, see the "dynamic\_profile dynamic collection" section in the *CANN Performance Tuning Tool*.
<!-- [dynamic\_profile dynamic collection](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0033.html#ZH-CN_TOPIC_0000002630711622__zh-cn_topic_0000002521150870_section17272160135118) -->

## Usage Constraints

- When no code is manually added to the script, this environment variable applies to TorchNPU training scenarios.
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
- <term>Ascend 950DT</term>
