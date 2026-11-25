# TORCH\_NPU\_FXRT\_EXECUTOR

## 功能描述

通过此环境变量指定FXRT图运行时的执行器。该配置同时适用于两种入口：Inductor的`options={"fxrt": True}`和`torch.compile(..., backend="fxrt")`。具体请参见[FXRT](../../../user_guide/torch_compile/compilation_backend/fxrt.md)。

- 未设置，或配置为`default`：使用FXRT默认执行器。
- 配置为`GERT`：使用`GERT`执行器，要求CANN版本不低于9.3.0。

> [!NOTE]
>
> 执行器在每个图编译时读取。请在编译对象首次执行前设置。合法取值只有`default`和`GERT`，其他取值会报错。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

使用默认执行器。可以不设置该环境变量，不设置时走默认运行时：

```bash
# 可以不设置。不设置时走默认运行时。
export TORCH_NPU_FXRT_EXECUTOR=default
```

使用GERT执行器：

```bash
export TORCH_NPU_FXRT_EXECUTOR=GERT
```

## 使用约束

- 仅在启用FXRT时生效。Inductor编译需设置`fxrt=True`，或指定`torch.compile()`的`backend`为`"fxrt"`。
- 配置为`GERT`时，要求CANN版本不低于9.3.0。合法取值只有`default`和`GERT`，其他取值会报错。

## 支持的型号

<!-- npu="950" id1 -->
<term>Ascend 950DT系列产品</term>
<!-- end id1 -->
