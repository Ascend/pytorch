# PROF\_CONFIG\_PATH

## Feature Description

In TorchNPU training scenarios, this environment variable specifies the path to the profiler_config.json configuration file of the dynamic_profile collection function of the TorchNPU Profiler interface.

## Configuration Example

```bash
export PROF_CONFIG_PATH="/path/to/profiler_config_path"
```

- After this environment variable is configured and training is started, `dynamic_profile` automatically creates the template file `profiler_config.json` under `profiler_config_path`. You can customize the configuration items based on the template file.
- The path specified by `PROF_CONFIG_PATH` can be customized (read and write permissions are required). The path format supports only strings consisting of letters, digits, and underscores. Soft links are not supported. For example, `/home/xxx/profiler_config_path`.
- For details about the dynamic_profile collection function and the profiler_config.json file, see the "dynamic\_profile dynamic collection" section in the *CANN Performance Tuning Tool*.
<!-- [dynamic\_profile dynamic collection](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0033.html#ZH-CN_TOPIC_0000002630711622__zh-cn_topic_0000002521150870_section17272160135118) -->

## Usage Constraints

- If no code is manually added to the script, this environment variable applies to TorchNPU training scenarios.
- After the `dynamic_profile` module is added to the script, this environment variable can be used in non-training scenarios. For example:

    ```python
    # Load the dynamic_profile module
    from torch_npu.profiler import dynamic_profile as dp
    # Set the path of the profiling configuration file
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
