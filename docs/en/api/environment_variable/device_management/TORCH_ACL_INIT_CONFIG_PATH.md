# TORCH\_ACL\_INIT\_CONFIG\_PATH

## Function Description

This environment variable specifies the path of the aclInit initialization configuration file, which is used to pass a custom ACL JSON configuration during the NPU initialization phase, replacing the default `acl_default.json` or `acl.json`.

- When configured with a valid file path: the user-specified configuration file is used for ACL initialization.
- When not configured or the path is invalid: the default configuration file is used (`acl_default.json` for CANN >= 8.3.RC1; otherwise, `acl.json`).

This environment variable is not configured by default.

## Configuration Example

```bash
export TORCH_ACL_INIT_CONFIG_PATH=/path/to/custom_acl.json
```

`acl_default.json` Special Notes

With CANN >= 8.3.RC1, the default configuration file `acl_default.json` contains the `defaultDevice` configuration:

```json
{
    "dump": {"dump_scene": "lite_exception"},
    "defaultDevice": {"default_device": "0"}
}
```

The `defaultDevice` configuration is used for the default Device configuration (used to configure the default compute device). The set_device operation is not executed immediately during the initialization phase, so that the set_device operation is executed lazily on demand.

When a user customizes the configuration file:

- **CANN >= 8.3.RC1**: The user configuration file must contain `"defaultDevice": {"default_device": "0"}`; otherwise, an error is reported.
- **CANN < 8.3.RC1**: The user configuration file should not contain `defaultDevice`; otherwise, an error is reported.

For the supported detailed configurations, see the "[aclInit](https://gitcode.com/cann/runtime/blob/9.2.0/docs/en/api_ref/02_initialization_and_deinitialization.md#aclinit)" section in the *CANN Runtime API*.

## Usage Constraints

- This environment variable must be set before NPU initialization; otherwise, it does not take effect.
- The configured path must be valid, and the configuration file must be in valid JSON format.

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
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>
<!-- end id4 -->
