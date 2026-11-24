# TORCH\_HCCL\_DESYNC\_DEBUG

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制是否进行通信超时分析。默认值为0。

- 配置为“0”：不开启通信超时分析。

- 配置为“1”：开启通信超时分析。

该变量记录各rank通信开始和结束的进度，供WatchDog在检查到Work超时时尝试生成Desync报告，但其本身不决定错误发生后是否终止进程。

非阻塞等待模式下，若开启Desync诊断且异步错误处理模式为0，进程组初始化时会将自身策略从模式NoHandling 提升为模式SkipCleanUp ，即WatchDog处理Work异常时跳过该分支的通信器清理并抛出异常——默认重抛配置下，相关Work异常仍可能终止进程。

此外，TORCH_DISTRIBUTED_DEBUG=DETAIL 也会启用相关Desync诊断，因此该变量为0不代表所有组合下诊断均关闭。

> [!NOTE]
>
> - 当前版本同时兼容旧名称`HCCL_DESYNC_DEBUG`。
> - 当`TORCH_HCCL_DESYNC_DEBUG`和`HCCL_DESYNC_DEBUG`同时配置时，优先使用`TORCH_HCCL_DESYNC_DEBUG`。
> - 应在进程组创建前配置，建议在各Rank启动前统一设置。
> - Desync报告依赖Store中的进度记录。Store不可用或部分rank提前退出时，可能无法生成完整报告，不能替代各rank原始日志的分析。

## 配置示例

推荐配置方式如下：

```bash
export TORCH_HCCL_DESYNC_DEBUG=1
```

兼容旧名称的配置方式如下：

```bash
export HCCL_DESYNC_DEBUG=1
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
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id5 -->
