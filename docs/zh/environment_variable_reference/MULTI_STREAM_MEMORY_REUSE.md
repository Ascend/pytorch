# MULTI\_STREAM\_MEMORY\_REUSE

## 功能描述

通过此环境变量可配置多流内存复用是否开启。集合通信多流场景，对Ascend Extension for PyTorch多流内存管理做优化，避免集合通信输入输出内存在多流场景下延迟释放，降低内存峰值。

-   0：关闭内存复用。
-   1：开启内存复用，基于eraseStream的方式，把之前的recordStream标记进行擦除，保证内存复用，持有tensor的弱引用，不延长tensor的生命周期。
-   2：开启内存复用，基于不执行recordStream标记的方法，保证内存复用能力，持有tensor的强引用，可能延长tensor的生命周期，当前不推荐使用。
-   3：开启内存复用，基于设置值为“1”做了进一步复用优化，可以在tensor提前释放的场景下，擦除recordStream标记。

默认值为1。

## 配置示例

```
export MULTI_STREAM_MEMORY_REUSE=0
```

## 使用约束

无

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 800I A2 推理产品</term>

