# PYTORCH\_NO\_NPU\_MEMORY\_CACHING

## 功能描述

通过此环境变量可配置是否关闭内存复用机制。

-   未配置时，开启内存复用机制。
-   配置为“1”时，关闭内存复用机制。

此环境变量默认为未配置。

关闭内存复用机制后，每次申请内存通过`aclrtmalloc`接口，生命周期结束后，立即通过`aclrtfree`接口释放回驱动。

> [!CAUTION]  
> -   关闭内存复用机制，默认使用`aclrtmalloc`和`aclrtfree`接口，虚拟内存默认关闭。
> -   关闭内存复用机制，作为一种debug手段，配置后模型性能可能会下降，在内存申请、释放频繁的模型场景可能出现严重下降。

## 配置示例

关闭内存复用机制示例：

```
export PYTORCH_NO_NPU_MEMORY_CACHING=1
```

重新启用内存复用机制示例：

```
unset PYTORCH_NO_NPU_MEMORY_CACHING
```

## 使用约束

若需使用torch\_npu.npu.check\_uce\_in\_memory，此环境变量必须为未配置状态，即开启内存复用机制。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 推理系列产品</term>

