# \(beta\) ACL\_STREAM\_TIMEOUT

> [!NOTICE]  
> 此功能尚在实验阶段，请谨慎使用。

## 功能描述

通过此环境变量可配置NPU流同步的超时等待时间，若流上任务在所设置的超时时间内未完成，同步接口将返回超时错误，避免进程因任务异常而无限期挂起。

超时时间单位为ms，取值说明如下：
- -1：表示永久等待，直至流上任务完成。
- \[1, 2147483647\]：表示具体的超时时间。

默认值为-1。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export ACL_STREAM_TIMEOUT=300000
```

## 使用约束

- 该环境变量依赖ACL运行时接口aclrtSynchronizeStreamWithTimeout。
- 该环境变量仅在CANN 6.0.1及以上版本生效；若版本不满足，流同步将回退至无限等待模式，此时该环境变量无效。
- 该环境变量仅作用于流同步的超时等待；设备级同步的超时时间请参见[ACL\_DEVICE\_SYNC\_TIMEOUT](ACL_DEVICE_SYNC_TIMEOUT.md)，两者配置相互独立。

## 支持的型号

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id3 -->
