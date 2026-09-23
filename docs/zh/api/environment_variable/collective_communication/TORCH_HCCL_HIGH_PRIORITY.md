# TORCH\_HCCL\_HIGH\_PRIORITY

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制是否强制从高优先级NPU stream pool中获取通信流。

- 配置为“0”：不强制使用高优先级stream，PG option中的优先级设置可能生效。
- 配置为“1”：强制使用高优先级NPU stream pool中的通信流。

该环境变量默认值为“0”。

> [!NOTE]
>
> - 当前版本同时兼容旧名称`HCCL_HIGH_PRIORITY`。
> - 高优先级stream可能减少通信延迟，但可能影响计算与通信的并发调度。

该变量对应PyTorch的[`TORCH_NCCL_HIGH_PRIORITY`](https://docs.pytorch.org/docs/2.14/torch_nccl_environment_variables.html)，配置方式一致，默认均关闭。

## 配置示例

```bash
export TORCH_HCCL_HIGH_PRIORITY=1
```

## 使用约束

无

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
