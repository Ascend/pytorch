# ACLNN\_EXTENSION\_SWITCH

## 功能描述

通过此环境变量可控制OpPlugin代码生成（codegen）过程中是否启用ACLNN扩展代码路径。

- 配置为`true`或`ON`时：开启该功能，torchnpugen工具在生成算子代码时使用ACLNN扩展相关的源码路径、模板和注册方式；
- 未配置时：不使用该功能，使用内置的默认代码生成逻辑。

该环境变量默认未配置，此时不启用ACLNN扩展代码路径。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export ACLNN_EXTENSION_SWITCH=true
```

## 使用约束

- 仅在代码生成（codegen）阶段生效，不影响运行时行为。
- 需与[ACLNN\_EXTENSION\_PATH](ACLNN_EXTENSION_PATH.md)配合使用，开启扩展时需同时指定有效的扩展代码路径。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
