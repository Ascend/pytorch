# P2P\_HCCL\_BUFFSIZE

## 功能描述

通过此环境变量可配置是否开启点对点通信（torch.distributed.isend、torch.distributed.irecv和torch.distributed.batch\_isend\_irecv）使用独立通信域功能。

-   配置为0时：关闭点对点通信使用独立通信域功能。
-   配置大于等于1时：开启点对点通信使用独立通信域功能，并且缓存区大小为配置值。

单位为MB，默认配置为20。

当开启点对点通信使用独立通信域功能时，每一个通信域都会额外占用2\*P2P\_HCCL\_BUFFSIZE大小的缓存区。若集群网络中存在较多的通信域，此缓存区占用量就会增多，可能存在影响模型数据正常存放的风险，此种场景下，可通过此环境变量减少点对点通信域占用的缓存区大小；若业务的模型数据量较小，但点对点通信数据量较大，则可通过此环境变量增大点对点通信域占用的缓存区大小，提升点对点通信效率。

## 配置示例

```
export P2P_HCCL_BUFFSIZE=20
```

## 使用约束

-   该环境变量申请的内存为HCCL独占，不可与其他业务内存复用。
-   每个通信域额外占用“2\*P2P\_HCCL\_BUFFSIZE”大小的内存，分别用于收发内存。
-   该资源按通信域粒度管理，每个通信域独占一组“2\*P2P\_HCCL\_BUFFSIZE”大小的内存。
-   Ascend Extension for PyTorch 7.1.0版本此环境变量已默认配置为20MB，若升级后出现oom，可在模型侧将此环境变量配置为0。
-   若之前未对P2P创建独立通信域，配置该环境变量后，会对P2P独立创建通信域，若模型侧存在send和recv下发间隔时间长的场景，会出现超时的可能，需要将HCCL\_CONNECT\_TIMEOUT的时间配置得更长，建议配置值为600s，具体配置值需依据模型脚本实际情况，HCCL\_CONNECT\_TIMEOUT具体可参考《CANN 环境变量参考》中的“[HCCL\_CONNECT\_TIMEOUT](https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0077.html)”章节。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>

