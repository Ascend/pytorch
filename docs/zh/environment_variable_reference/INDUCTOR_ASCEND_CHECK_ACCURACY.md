# （beta）INDUCTOR\_ASCEND\_CHECK\_ACCURACY

## 功能描述

INDUCTOR_ASCEND_CHECK_ACCURACY是Ascend Extension for PyTorch提供的精度校验工具，在torch.compile图编译后端为“inductor”时自动检测融合算子的数值精度。

该工具可捕获融合算子对应的FX子图，生成独立可执行的单算子测试用例，并在相同输入条件下比对eager与融合算子的输出差异。当差异超出预设阈值时，自动输出精度校验失败日志及诊断信息，辅助开发者快速定位精度问题。

## 配置示例

示例一：启用精度工具，使用默认精度阈值配置

```bash
export INDUCTOR_ASCEND_CHECK_ACCURACY=1
```

**表 1** 默认精度阈值

| 数据类型 | 相对误差rtol | 绝对误差atol |
|:---:|:---:|:---:|
| float32 | 1.3e-6 | 1e-5 |
| float16 | 1e-3 | 1e-5 |
| bfloat16 | 1.6e-2 | 1e-5 |
| 其他 | 1.3e-6 | 1e-5 |

示例二：启用精度工具，并设置精度对比阈值

```bash
export INDUCTOR_ASCEND_CHECK_ACCURACY=1
# 设置精度对比时相对误差阈值为1e-6，绝对误差阈值为1e-7
export INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL="rtol=1e-6,atol=1e-7"
```

> [!CAUTION]
>
> 如需根据不同数据类型（如float32、float16、bfloat16等）配置不同的精度阈值，请手动修改不同后端源码中的acc_comp_tol字典。各后端配置文件路径如下：
>
> - Triton：[config](../../../torch_npu/_inductor/config.py)
> - MLIR和DVM：[config](../../../torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/config.py)

精度比对无异常时不打印额外信息。比对有异常时打印如下格式的告警：

```bash
CHECK ACCURACY FAILED! Kernel: <kernel_name>, Output idx: <idx>, Mismatched: <m>/<n> (<pct>%), Greatest Rel Diff: <rel>, Greatest Abs Diff: <abs>, rtol: <rtol>, atol: <atol>, dump_path: <path> 
```

当设置为Triton后端时会将数据及fx graph文件保存至打印信息的dump_path目录下。

## 使用约束

- 此环境变量仅可在PyTorch2.7.1和PyTorch2.9.0版本使用。

- 在torch.compile图编译后端为“inductor”时可使用此环境变量。

- 此环境变量开启时,Triton后端会自动将`torch._inductor.config.split_reductions`设置为`False`。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
