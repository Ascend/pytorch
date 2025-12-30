# PER\_STREAM\_QUEUE

> [!NOTICE]  
> 本特性当前为试验性功能，后续版本可能存在变更。

## 功能描述

通过此环境变量可配置是否开启一个stream一个task\_queue算子下发队列。

-   配置为“0“时，关闭一个stream一个task\_queue算子下发队列。
-   配置为“1“时，开启一个stream一个task\_queue算子下发队列。

此环境变量默认配置为“0“。

## 配置示例

```
export PER_STREAM_QUEUE=1
```

## 使用约束

当[TASK\_QUEUE\_ENABLE](TASK_QUEUE_ENABLE.md)配置为“1“/“2“时，此环境变量才能生效。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 800I A2 推理产品</term>
-   <term>Atlas 推理系列产品</term>

