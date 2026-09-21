# SHUT\_DOWN\_FX\_PASS\_LIST

## 功能描述

通过此环境变量可指定需要关闭的`torch_npu`图优化FX pass列表。列表中的pass在注册阶段即被跳过（不参与推理与训练的图优化），用于规避特定pass引入的问题或对照验证pass效果。

- 配置为`all`：关闭全部自定义FX pass。
- 配置为逗号分隔的pass名称列表：列表中的pass不注册。

> [!NOTE]
>
> - 该环境变量需在导入`torch_npu`之前设置。
> - pass名称为注册时的函数名，具体命名规范请参考[ascend_graph_pass.py](../../../../../torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py)。
> - 该变量在默认关闭列表的基础上追加，不会打开已默认关闭的pass。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

关闭指定pass：

```bash
export SHUT_DOWN_FX_PASS_LIST="view_fold_pass,fold_cat"
```

关闭全部自定义pass：

```bash
export SHUT_DOWN_FX_PASS_LIST="all"
```

## 使用约束

无

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas 训练系列产品</term>
<!-- end id1 -->
<!-- npu="950" id2 -->
- <term>Ascend 950DT</term>
<!-- end id2 -->
