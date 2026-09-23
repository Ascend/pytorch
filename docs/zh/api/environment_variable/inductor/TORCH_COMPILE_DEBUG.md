# TORCH\_COMPILE\_DEBUG

## 功能描述

通过此环境变量可开启`torch.compile`的调试模式。开启后，系统会在编译过程中导出FX图、codegen输出等调试信息，便于开发者分析编译流程和定位问题。

- 默认值为“0”：不开启调试模式。
- 配置为“1”：开启调试模式，导出Dynamo输出图、codegen生成的kernel代码等。

> [!NOTE]
>
> - 该环境变量需在进程启动前配置，进程运行过程中修改不会生效。
> - 开启调试模式后，编译产物（FX图、output_code.py等）会保存到`torch_compile_debug/`目录下。
> - 调试模式会产生额外的I/O开销，建议仅在问题定位时使用。

该变量对应PyTorch的[TORCH_COMPILE_DEBUG](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting_old.html)，配置方式一致。

## 配置示例

```bash
export TORCH_COMPILE_DEBUG=1
```

## 使用约束

无

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id4 -->
