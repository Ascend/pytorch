# TORCH\_HCCL\_ENABLE\_TIMING

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制是否为通信操作启用计算设备执行时长的NPU计时事件。

- 0：不通过该变量启用计时事件。
- 1：启用计时事件，用于在操作完成且相关信息可用时生成FR记录中的`duration_ms`。

默认值：0。

> [!NOTE]
>
> 开启`TORCH_HCCL_DESYNC_DEBUG`也会启用计时。因此，该变量为0不表示在所有配置组合下都关闭计时。

## 配置示例

```bash
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_HCCL_ENABLE_TIMING=1
```

## 使用约束

- 应在提交通信操作前设置，建议在启动训练进程前统一配置。
- 未开启计时时仍可记录通信操作、张量形状及状态等信息；无法计算时长时，`duration_ms`可能不存在，不应将缺失值视为0。
- 当操作未完成、记录未归档或相关事件不可用时，即使已启用计时功能，部分记录可能不包含`duration_ms`字段。
- 该变量不启用FR缓冲区、心跳监控或自动转储。完整使用方法请参见[飞行记录器](../../../developer_notes/fault_diagnosis/flight_recorder.md)。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id4 -->
