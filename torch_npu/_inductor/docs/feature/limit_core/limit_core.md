# NPU_DEVICE_LIMIT

## 功能描述

用于将NPU卡的计算核（Cube Core、Vector Core）进行划分，例如：通过export NPU_DEVICE_LIMIT='14,28'，将会划分14个Cube Core和28个Vector Core作为当前可用的计算资源。在这种情况下，一个计算图中所涉及的算子(AclNN算子、triton手写算子、triton自动融合算子、catlass算子)，最多可以使用这些受限的计算核。

适用于小shape模型场景，算子shape较小，打不满cube core和vector core导致浪费。因此，通过分核/限核，支持多个实例同时推理，即多个算子同时运行在同一个NPU卡上。

| 值 | 说明 |
|---|---|
| 例'7,14'或者'14,28' | Cube和Vector的核数限制 |

## 配置示例

```bash
export NPU_DEVICE_LIMIT='14,28'  
```

## 使用约束
- 在A2/A3/A5代际，Cube和Vector的配比是1:2。因此，设置NPU_DEVICE_LIMIT时，建议Cube和Vector的数量比例达成1:2。
- 如不设置，则默认使用NPU上全部的Cube和Vector核；

## 支持的型号

- <term>Atlas A5 系列产品</term>
