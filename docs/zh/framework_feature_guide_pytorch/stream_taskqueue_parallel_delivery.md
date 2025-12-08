# Stream级TaskQueue并行下发

> [!NOTICE]  
> 本特性当前为试验性功能，后续版本可能存在变更。

## 简介

当前开启task\_queue算子下发队列时采用单Device单TaskQueue模式，所有Stream共享同一个任务队列。一级流水线程（多线程）向统一队列提交任务，二级流水线程从队列中串行取出任务下发。这种架构在高并发场景下（多Stream同时提交）会出现队列竞争问题，导致任务下发存在串行化瓶颈。

为解决此问题，Ascend Extension for PyTorch推出Stream级TaskQueue并行下发特性。开启该特性后，每个Stream会初始化独立的TaskQueue和对应的Dequeue线程，实现真正的二级流水并行下发机制。不同Stream的任务可以并行下发，有效解决了队列竞争问题，提升了高并发场景下的任务下发效率。

## 使用场景

多线程多流下发，dequeue成为阻塞时，推荐使用该特性。

## 使用指导

通过PER\_STREAM\_QUEUE环境变量可配置是否开启一个stream一个task\_queue算子下发队列。

-   配置为“0“时，关闭一个stream一个task\_queue算子下发队列。
-   配置为“1“时，开启一个stream一个task\_queue算子下发队列。

此环境变量默认配置为“0“。

## 使用样例

```
export PER_STREAM_QUEUE=1
```

## 约束说明

-   该特性依赖taskqueue，当TASK\_QUEUE\_ENABLE配置为“1“/“2“时，此此特性才能生效。
-   该特性不支持快恢场景。
-   开启此特性时，非默认流的taskqueue的oom不会立即触发内存快照。
-   开启此特性时，多流情况下会有多个taskqueue，对应多个线程，可能存在资源抢占，影响性能。
-   开启此特性可能存在多个二级流水线程，不支持细粒度绑核功能。

