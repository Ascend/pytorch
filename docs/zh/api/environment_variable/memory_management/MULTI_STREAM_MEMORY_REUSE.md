# MULTI\_STREAM\_MEMORY\_REUSE

## 功能描述

通过此环境变量可配置多流内存复用是否开启。在集合通信多流场景，对TorchNPU多流内存管理做优化，避免集合通信输入输出内存在多流场景下延迟释放，降低内存峰值。

- 0：关闭内存复用。
- 1：开启内存复用，基于eraseStream的方式，把之前的recordStream标记进行擦除，保证内存复用，持有tensor的弱引用，不延长tensor的生命周期。
- 2：开启内存复用，基于不执行recordStream标记的方法，保证内存复用能力，持有tensor的强引用，可能延长tensor的生命周期。
- 3：开启内存复用，在设置值为“1”的基础上做了进一步复用优化，可以在tensor提前释放的场景下，擦除recordStream标记。

默认值为2。

## 配置示例

```bash
export MULTI_STREAM_MEMORY_REUSE=0
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
<!-- npu="910b" id5 -->
- <term>Atlas 800I A2训练服务器</term>
<!-- end id5 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id4 -->
