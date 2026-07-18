# PROF_CONFIG_PATH

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:05:20.798Z pushedAt=2026-06-16T03:14:22.329Z -->

## Function Description

In PyTorch training scenarios, this environment variable specifies the path to the profiler_config.json configuration file for the dynamic_profile collection feature of the Ascend PyTorch Profiler interface.

## Configuration Example

```bash
export PROF_CONFIG_PATH="/path/to/profiler_config_path"
```

- After configuring the environment variable and starting training, `dynamic_profile` will automatically create a template file `profiler_config.json` under `profiler_config_path`. You can customize the configuration items based on the template file.
- The path specified by `PROF_CONFIG_PATH` can be customized (read and write permissions are required). The path format only supports strings consisting of letters, digits, and underscores. Soft links are not supported, for example, `/home/xxx/profiler_config_path`.
- For a detailed introduction to the dynamic_profile feature and the `profiler_config.json` file, see the [dynamic_profile](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0033.html)" section in the *CANN Performance Tuning Tool*.

## Usage Constraints

- In scenarios where no code is manually added to the script, this environment variable applies to PyTorch training scenarios.
- After the dynamic_profile module is added to the script, this environment variable can be used in non-training scenarios. For example:

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
