# PYTORCH\_NO\_NPU\_MEMORY\_CACHING

## 功能描述

通过此环境变量可配置是否关闭内存复用机制。

- 未配置或配置为“0”时，开启内存复用机制。
- 配置为“1”时，关闭内存复用机制。

此环境变量默认为未配置。

关闭内存复用机制后，每次通过`aclrtMallocAlign32`或`aclrtMalloc`接口申请内存，生命周期结束后，立即通过`aclrtFree`接口释放回驱动。

> [!CAUTION]  
>
> - 关闭内存复用机制，默认使用`aclrtMalloc`和`aclrtFree`接口，虚拟内存默认关闭。
> - 关闭内存复用机制，作为一种debug手段，配置后模型性能可能会下降，在内存申请、释放频繁的模型场景下，性能可能出现明显下降。

## 配置示例

关闭内存复用机制示例：

```bash
export PYTORCH_NO_NPU_MEMORY_CACHING=1
```

重新启用内存复用机制示例：

```bash
unset PYTORCH_NO_NPU_MEMORY_CACHING
# 或
export PYTORCH_NO_NPU_MEMORY_CACHING=0
```

## 使用约束

若需使用torch\_npu.npu.check\_uce\_in\_memory，此环境变量必须为未配置状态，即开启内存复用机制。

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
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT</term>
<!-- end id5 -->
