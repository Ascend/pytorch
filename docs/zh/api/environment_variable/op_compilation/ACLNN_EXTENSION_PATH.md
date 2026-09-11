# ACLNN\_EXTENSION\_PATH

## 功能描述

通过此环境变量可指定ACLNN扩展代码的搜索路径。在op-plugin代码生成（codegen）过程中，torchnpugen工具根据此路径查找ACLNN扩展相关的op\_plugin源文件、自定义算子YAML配置和exposed\_api.py等文件。通常与[ACLNN\_EXTENSION\_SWITCH](ACLNN_EXTENSION_SWITCH.md)配合使用，当ACLNN扩展开关开启时，此路径生效。

此环境变量默认不配置，此时torchnpugen使用内置的`third_party/op-plugin`目录作为默认搜索路径。

## 配置示例

```bash
export ACLNN_EXTENSION_PATH=/path/to/aclnn/extension
```

> [!NOTE]
>
> 此功能为`torch_npu`特有，PyTorch社区无直接对应变量。

## 使用约束

- 仅在代码生成（codegen）阶段生效，不影响运行时行为。
- 需与[ACLNN\_EXTENSION\_SWITCH](ACLNN_EXTENSION_SWITCH.md)配合使用，单独设置不生效。
- 路径需包含有效的op\_plugin目录和相关源文件。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
