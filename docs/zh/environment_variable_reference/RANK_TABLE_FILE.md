# RANK\_TABLE\_FILE

## 功能描述

通过此环境变量可配置是否通过RANK\_TABLE\_FILE进行集合通信域建链。

-   未配置时，通过默认的协商流程进行集合通信域建链。
-   配置且文件全路径有效时，通过RANK\_TABLE\_FILE进行集合通信域建链。

此环境变量默认未配置。

> [!NOTE]  
> 配置RANK\_TABLE\_FILE场景下，执行模型分布式训练时如果出现“RuntimeError: The Inner Error ...”的报错，建议将HCCL\_CONNECT\_TIMEOUT的超时时间适当增大，避免ranktable场景下无协商导致的建链超时问题，具体请参考[在进行模型分布式训练时遇到报错“RuntimeError: The Inner Error ...”](runtimeerror_Inner_Error.md)。

## 配置示例

启用ranktable文件方式建链示例：

```
export RANK_TABLE_FILE=/home/ranktable.json
```

> [!CAUTION]  
> -   配置的文件路径不存在时，会通过默认的协商流程进行集合通信域建链。
> -   配置的文件路径存在，但配置信息有误时，不会通过默认的协商流程进行集合通信域建链，而是在实际通信时会进行相应的报错。
> -   配置的文件路径不能为软链接，且存在读取权限。
> -   配置的文件需要为json格式，具体可参考《[HCCL集合通信库用户指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hcclug/hcclug_000001.html)》中对应的“rank table配置资源信息”章节。

关闭ranktable文件方式建链示例：

```
unset RANK_TABLE_FILE
```

## 使用约束

无

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>

