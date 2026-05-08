# NPU_INDUCTOR_FALLBACK_LIST

## 功能描述

通过此环境变量可指定需要回退到PyTorch原生的算子列表。当某些算子在NPU后端执行出现问题或需要使用算子原生实现时，可通过此环境变量进行配置。

- 默认值为`None`，不进行算子回退。
- 配置指定算子：以逗号分隔的算子名称列表（如`aten.div,aten.add.Tensor`），对应的算子将回退到PyTorch原生实现。
- 配置为`allfallback`：所有算子都回退到PyTorch原生实现。

## 配置示例

- 指定特定算子进行回退：

```python
import os
os.environ["NPU_INDUCTOR_FALLBACK_LIST"] = "aten.div,aten.add.Tensor"
```

- 指定全部算子回退：

```python
import os
os.environ["NPU_INDUCTOR_FALLBACK_LIST"] = "allfallback"
```

## 使用约束

- 该环境变量仅在Inductor后端场景下生效。
- 算子名称需使用完整的aten算子命名格式（如`aten.div.Tensor`、`aten.add.Tensor`）。
- 多算子使用逗号进行分隔，不支持空格。
- 回退操作会导致相应算子失去NPU硬件加速能力，请谨慎使用。
- 建议仅在调试或排查问题时启用，生产环境不建议使用。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 800I A2 推理产品</term>
