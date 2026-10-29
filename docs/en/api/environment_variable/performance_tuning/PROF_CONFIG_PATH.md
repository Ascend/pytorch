# PROF\_CONFIG\_PATH

## Function Description

In TorchNPU training scenarios, this environment variable specifies the path to the profiler_config.json configuration file of the dynamic_profile collection function of the TorchNPU Profiler interface.

## Configuration Example

```bash
export PROF_CONFIG_PATH="/path/to/profiler_config_path"
```

- After this environment variable is configured and training is started, `dynamic_profile` automatically creates the template file `profiler_config.json` under `profiler_config_path`. You can customize the configuration items based on the template file.
- The path specified by `PROF_CONFIG_PATH` can be customized (read and write permissions are required). The path format supports only strings consisting of letters, digits, and underscores. Soft links are not supported. For example, `/home/xxx/profiler_config_path`.
- For details about the dynamic_profile collection function and the profiler_config.json file, see "[dynamic\_profile dynamic profiling](https://gitcode.com/Ascend/pytorch/blob/master/docs/en/developer_notes/ascend_pytorch_profiler_user_guide.md#profiling-and-parsing-performance-data-dynamic-profile)".

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
