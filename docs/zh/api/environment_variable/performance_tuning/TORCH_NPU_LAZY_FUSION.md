# TORCH\_NPU\_LAZY\_FUSION

## 功能描述

通过此环境变量可开启TorchNPU Eager模式下的DVM算子融合。DVM将多个相邻的小算子合并成单个融合kernel，减少kernel下发次数和中间张量的搬运，从而加速训练和推理。

- 配置为“True”时：开启DVM算子融合。
- 未配置或配置为“False”时：关闭DVM算子融合。

此环境变量默认为未配置。

> [!NOTE]
>
> 此环境变量用于Eager模式下的算子融合，无需经过图编译。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export TORCH_NPU_LAZY_FUSION=True
```

### 在脚本中按需开启融合

环境变量需在 `import torch_npu` 前设置。开启环境变量后，可以在用户脚本中按代码区间控制无图融合。

例如，临时关闭某个代码段的无图融合，退出后恢复之前的状态：

```python
from torch_npu.npu import lazy_fusion

with lazy_fusion.disabled():
    output = model_part(input)
```

也可以在外层关闭融合，仅在指定代码段中按需开启：

```python
from torch_npu.npu import lazy_fusion

with lazy_fusion.disabled():
    x = preprocess(input)
    with lazy_fusion.enabled():
        x = model_part(x)
    output = postprocess(x)
```

也可以直接设置开关，设置后持续生效，直到再次修改：

```python
from torch_npu.npu import lazy_fusion

lazy_fusion.set_disable()
x = preprocess(input)

lazy_fusion.set_enable()
output = model(x)
```

`set_enable()` 开启脚本侧融合开关，`set_disable()` 关闭脚本侧融合开关，两个接口均不接受参数，返回值为 `None`。与 `with` 不同，直接设置不会自动恢复；需要异常退出时自动恢复状态的代码段应使用上下文接口。

两种方式可混用：上下文退出时，将恢复进入前的状态（忽略上下文内部调用的 `set_enable()` 或 `set_disable()`）。

- 环境变量是总开关；未开启环境变量时，`lazy_fusion.enabled()` 和 `lazy_fusion.set_enable()` 均不能单独开启无图融合。
- 上下文支持嵌套，正常退出或区间内发生异常退出时均恢复之前的状态。
- 开关状态变化时会提交当前待融合图，避免算子跨越开启/关闭边界融合；接口不等待设备执行完成。
- 开关是进程级的，应在代码区间边界调用，切换时不要并发构建融合图。

## 使用约束

- 需在导入`torch_npu`之前设置。
- 仅在[TASK_QUEUE_ENABLE](../op_execution/TASK_QUEUE_ENABLE.md)为1或2时生效，否则自动禁用算子融合。
- 仅在主线程及其反向线程生效，其它独立线程（如dataloader worker）自动禁用。
- 仅在DVM支持的芯片型号上生效。

## 支持的型号

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id2 -->
