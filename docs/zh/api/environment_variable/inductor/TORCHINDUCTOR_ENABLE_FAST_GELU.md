# TORCHINDUCTOR\_ENABLE\_FAST\_GELU

## 功能描述

通过此环境变量可控制GELU激活函数的decomposition实现。启用后，`aten.gelu`在图模式下降解为tanh近似公式（`0.5*x*(1+tanh(√(2/π)*(x+0.044715*x³)))`的sigmoid等价形式），用乘加与sigmoid替代`erf`算子，提升kernel执行性能。

- 默认值为“0”：使用默认decomposition（含`erf`）。
- 配置为“1”：GELU降解为tanh近似（fast gelu）实现。

> [!NOTE]
>
> - 该环境变量需在导入`torch_npu`之前设置。
> - tanh近似与`erf`精确实现存在数值差异，对精度敏感的训练任务建议先验证。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export TORCHINDUCTOR_ENABLE_FAST_GELU=1
```

## 使用约束

无

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
