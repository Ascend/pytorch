# TORCH\_HCCL\_WAIT\_TIMEOUT\_DUMP\_MILSEC

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可设置heartbeat monitor等待异步dump完成的最大时间。当发生超时触发dump时，monitor会等待其他rank的dump完成。超过此时间后，无论dump是否完成，monitor都会终止进程。

该环境变量的默认值为60000，单位为ms。

该变量对应PyTorch的`TORCH_FR_WAIT_TIMEOUT_DUMP_MILSEC`，PyTorch同时兼容[`TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC`](https://docs.pytorch.org/docs/2.14/torch_nccl_environment_variables.html)，PyTorch默认值为15000。

## 配置示例

```bash
export TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC=30000
```

## 使用约束

此环境变量仅在`TORCH_HCCL_ENABLE_MONITORING=1`时生效。

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
