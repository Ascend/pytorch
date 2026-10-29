# KINETO\_USE\_DAEMON

## Function Description

This environment variable is used to set whether to enable dynamic\_profile collection through msMonitor nputrace in training scenarios.

## Configuration Example

```bash
export KINETO_USE_DAEMON=1
```

For detailed usage, see "[dynamic\_profile dynamic profiling](https://gitcode.com/Ascend/pytorch/blob/master/docs/en/developer_notes/ascend_pytorch_profiler_user_guide.md#profiling-and-parsing-performance-data-dynamic-profile)".

## Usage Constraints

- When no code is manually added to the script, this environment variable applies to TorchNPU training scenarios.
- After the dynamic_profile module is added to the script, this environment variable can be used in non-training scenarios. For example:

    ```python
    # Load the dynamic_profile module
    from torch_npu.profiler import dynamic_profile as dp
    # Set the path of the Profiling configuration file
    dp.init("profiler_config_path")
    …
    for step in steps:
        train_one_step()
        # Split into steps
        dp.step()
    ```

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
<!-- npu="950" id4 -->
- <term>Ascend 950DT products</term>
<!-- end id4 -->
