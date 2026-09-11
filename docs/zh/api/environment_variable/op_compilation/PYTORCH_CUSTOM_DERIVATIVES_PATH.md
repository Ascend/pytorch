# PYTORCH\_CUSTOM\_DERIVATIVES\_PATH

## 功能描述

通过此环境变量可指定自定义算子自动微分（autograd）定义文件`derivatives.yaml`的路径。在op-plugin代码生成（codegen）流程中，该文件定义了NPU自定义算子的反向传播规则，供torchnpugen代码生成器在生成算子autograd代码时使用。

此环境变量默认不配置，此时torchnpugen使用内置的`third_party/op-plugin/op_plugin/config/`目录下对应PyTorch版本的`derivatives.yaml`作为默认路径。

> [!NOTE]
>
> 此功能为`torch_npu`特有，PyTorch社区无直接对应变量。

## 配置示例

```bash
export PYTORCH_CUSTOM_DERIVATIVES_PATH=/path/to/op-plugin/op_plugin/config/pytorch_v2.1/derivatives.yaml
```

## 使用约束

- 仅在代码生成（codegen）阶段生效，非运行时环境变量。
- 路径需指向有效的`derivatives.yaml`文件，文件格式需符合PyTorch自定义导数定义规范。
- 需与[ACLNN\_EXTENSION\_SWITCH](ACLNN_EXTENSION_SWITCH.md)配合使用，仅在ACLNN扩展开关开启时生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
