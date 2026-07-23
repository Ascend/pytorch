# TORCH\_ACL\_INIT\_CONFIG\_PATH

## 功能描述

通过此环境变量可指定aclInit初始化配置文件的路径，用于在NPU初始化阶段传入自定义ACL JSON配置，替代默认的`acl_default.json`或`acl.json`。

- 配置为有效文件路径时：使用用户指定的配置文件进行ACL初始化。
- 未配置或路径无效时：使用默认配置文件（CANN >= 8.3.RC1使用`acl_default.json`，否则使用`acl.json`）。

此环境变量默认不配置。

## 配置示例

```bash
export TORCH_ACL_INIT_CONFIG_PATH=/path/to/custom_acl.json
```

`acl_default.json`特殊说明

配套CANN >= 8.3.RC1之后，默认配置文件`acl_default.json`包含`defaultDevice`配置：

```json
{
    "dump": {"dump_scene": "lite_exception"},
    "defaultDevice": {"default_device": "0"}
}
```

`defaultDevice`配置用于默认Device配置（用于配置默认的计算设备），初始化阶段不会立即执行set_device操作, 从而实现按需延迟执行set_device操作。

用户自定义配置文件时：

- **CANN >= 8.3.RC1**：用户配置文件必须包含`"defaultDevice": {"default_device": "0"}`，否则会报错。
- **CANN < 8.3.RC1**：用户配置文件不应包含`defaultDevice`，否则会报错。

支持的详细配置参考CANN aclInit接口说明(https://www.hiascend.com/document/detail/zh/canncommercial/900/API/runtimeapi/aclcppdevg_03_0022.html)。

## 使用约束

- 此环境变量必须在NPU初始化之前设置，否则不生效。
- 配置的路径必须有效，配置文件必须为合法JSON格式。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>
