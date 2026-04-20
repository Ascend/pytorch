# TORCHINDUCTOR_NDDMA

## 功能描述

此功能启用Triton-Ascend load随路转置能力。在A2、A3代际上开启此功能对性能无影响。在A5代际，系统会通过底层NDDMA特性做转置加速，转置性能有明显增益。
默认值根据芯片版本自动设置：Atlas A5代际为"1"，其他代际为"0"。

| 值 | 说明 |
|---|---|
| "0" | 关闭NDDMA功能 |
| "1" | 开启NDDMA功能 |

## 配置示例

```bash
export TORCHINDUCTOR_NDDMA=1
```

## 使用约束

- 在Atlas A2和A3代际上开启此功能对性能无影响。
- 在Atlas A5代际上开启此功能可以显著提升需要转置的Triton算子性能。

## 支持的型号

-   <term>Atlas A5 系列产品</term>
