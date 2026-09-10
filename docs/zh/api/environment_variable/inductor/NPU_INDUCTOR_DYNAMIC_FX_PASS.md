# NPU\_INDUCTOR\_DYNAMIC\_FX\_PASS

## 功能描述

通过此环境变量可控制图优化FX pass是否使用动态shape（symbolic shape）模式。启用后，针对动态shape的图优化pass按符号shape语义执行；关闭后，这些pass回退到legacy的静态shape行为。

- 默认值为`1`，使用动态shape模式的FX pass。
- 配置为`0`、`false`或`off`：回退到静态shape行为。

> [!NOTE]
>
> - 主要用于动态shape场景下FX pass行为异常时的回退开关。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

回退静态行为：

```bash
export NPU_INDUCTOR_DYNAMIC_FX_PASS=0
```

## 使用约束

需在导入`torch_npu`之前设置。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
