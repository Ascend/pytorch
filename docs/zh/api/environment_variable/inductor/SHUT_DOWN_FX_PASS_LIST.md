# SHUT\_DOWN\_FX\_PASS\_LIST

## 功能描述

通过此环境变量可指定需要关闭的`torch_npu`图优化FX pass列表。列表中的pass在注册阶段即被跳过（不参与推理与训练的图优化），用于规避特定pass引入的问题或对照验证pass效果。

- 默认值未配置，仅内置默认关闭的pass保持关闭（如`fused_matmul_relu_pass`等，由对应config开关决定）。
- 逗号分隔的pass名称列表：列表中的pass不注册。
- 配置为`all`：关闭全部自定义FX pass。

> [!NOTE]
>
> - pass名称为注册时的函数名（如`fused_matmul_relu_pass`、`multi_slice_concat_pass`），可通过`torch_npu`日志或源码`torch_npu/_inductor/fx_passes/ascend_custom_passes/`确认。
> - 该变量在默认关闭列表的基础上追加，不会打开已默认关闭的pass。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

关闭指定pass：

```bash
export SHUT_DOWN_FX_PASS_LIST="fused_matmul_relu_pass,multi_slice_concat_pass"
```

关闭全部自定义pass：

```bash
export SHUT_DOWN_FX_PASS_LIST="all"
```

## 使用约束

需在导入`torch_npu`之前设置。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
