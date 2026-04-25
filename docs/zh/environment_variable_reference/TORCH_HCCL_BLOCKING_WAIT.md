# TORCH_HCCL\_BLOCKING\_WAIT

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制`ProcessGroupHCCL`中`wait()`和`synchronize()`的同步模式（阻塞或非阻塞）。

- `0`（默认值）：关闭阻塞等待。
- `1`：开启阻塞等待。


> [!NOTE]
>
> - 当前版本同时兼容旧名称`HCCL_BLOCKING_WAIT`。
> - 当`TORCH_HCCL_BLOCKING_WAIT`和`HCCL_BLOCKING_WAIT`同时配置时，优先使用`TORCH_HCCL_BLOCKING_WAIT`。
> - 开启此环境变量后，`wait()`和`synchronize()`会在主机侧等待本次HCCL通信真正完成、报错或超时后再返回。
> - 开启此环境变量后，不会创建 watchdog 线程。

## 配置示例

推荐配置方式如下：

```bash
export TORCH_HCCL_BLOCKING_WAIT=1
```

兼容旧名称的配置方式如下：

```bash
export HCCL_BLOCKING_WAIT=1
```

## 使用约束

- 此环境变量仅在使用HCCL作为通信后端时生效。
- 此环境变量在 `ProcessGroupHCCL` 初始化时读取；若在`init_process_group`或`new_group`创建进程组之后再修改，不会影响已创建的进程组。


## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>
