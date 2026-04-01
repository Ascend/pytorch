# （beta）INDUCTOR\_ASCEND\_CHECK\_ACCURACY

## 功能描述

INDUCTOR_ASCEND_CHECK_ACCURACY是Ascend Extension for PyTorch提供的精度校验工具，仅在torch.compile图编译后端为"Inductor"且模式为"Triton"时自动检测融合算子的数值精度。

该工具可捕获融合算子对应的FX子图，生成独立可执行的单算子测试用例，并在相同输入条件下比对eager与triton的输出差异。当差异超出预设阈值时，自动输出精度校验失败日志及诊断信息，辅助开发者快速定位精度问题。

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

> [!CAUTION] 注意<br>
如需根据不同数据类型（如float32、float16、bfloat16等）配置不同的精度阈值，请手动修改源码[config](../../../torch_npu/_inductor/config.py)中的acc_comp_tol字典。

## 使用约束

- 此环境变量仅可在PyTorch2.7.1版本使用。

- 在torch.compile图编译后端为"Inductor"且模式为"Triton"（环境变量TORCHINDUCTOR_NPU_BACKEND为空或者"default"）时可使用此环境变量。

## 支持的型号

- <term>Atlas A2训练系列产品</term>
- <term>Atlas A3训练系列产品</term>
