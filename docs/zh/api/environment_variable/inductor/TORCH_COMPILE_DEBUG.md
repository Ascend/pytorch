# TORCH\_COMPILE\_DEBUG

## 功能描述

通过此环境变量可开启`torch.compile`的调试模式。开启后，系统会在编译过程中导出FX图、codegen输出等调试信息，便于开发者分析编译流程和定位问题。

- 默认值为`0`，不开启调试模式。
- 配置为`1`：开启调试模式，导出Dynamo输出图、codegen生成的kernel代码等。

> [!NOTE]
>
> - 开启调试模式后，编译产物（FX图、output_code.py等）会保存到`torch_compile_debug/`目录下。
> - 调试模式会产生额外的I/O开销，建议仅在问题定位时使用。

该变量沿用PyTorch的同名环境变量，配置方式一致。

## 配置示例

```bash
export TORCH_COMPILE_DEBUG=1
```

## 使用约束

需在进程启动前配置，进程运行过程中修改不会生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
