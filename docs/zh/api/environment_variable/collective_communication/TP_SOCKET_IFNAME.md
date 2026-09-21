# TP\_SOCKET\_IFNAME

## 功能描述

通过此环境变量可指定TensorPipe RPC传输层使用的网络接口名称。当节点有多个网络接口时，用于绑定特定的网卡。

- 未配置或默认配置：通过hostname解析IP地址。
- 配置指定接口名：绑定指定网卡，通过`lookupAddrForIface`查找该接口的IP地址。

> [!NOTE]
>
> 如果指定接口查找失败会回退到`127.0.0.1`。

该变量沿用PyTorch的同名环境变量，配置方式一致。

## 配置示例

```bash
export TP_SOCKET_IFNAME=eth0
```

## 使用约束

无

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas 训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT</term>
<!-- end id4 -->
