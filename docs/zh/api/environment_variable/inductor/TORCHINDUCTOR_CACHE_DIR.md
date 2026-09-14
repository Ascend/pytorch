# TORCHINDUCTOR\_CACHE\_DIR

## 功能描述

通过此环境变量可配置Inductor编译缓存的目录路径。设置后，Inductor会将编译生成的kernel缓存到指定目录，便于后续复用，加速后续编译过程。

- 默认值未配置：使用默认缓存目录（通常为系统临时目录下的`torchinductor_<username>`）。
- 配置为目录路径：将编译产物缓存到指定目录。

> [!NOTE]
>
> - 该环境变量需在导入`torch_npu`之前设置。
> - 需确保目标目录可写且有足够的磁盘空间。
> - 此环境变量在全部模式（`TORCHINDUCTOR_NPU_BACKEND="default"`/`"dvm"`/`"ascendc"`等）下均生效。
> - 设置独立的缓存目录便于多用户共享缓存或持久化缓存。

该变量对应PyTorch的[TORCHINDUCTOR_CACHE_DIR](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html)，配置方式一致。

## 配置示例

```bash
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
```

## 使用约束

无

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
