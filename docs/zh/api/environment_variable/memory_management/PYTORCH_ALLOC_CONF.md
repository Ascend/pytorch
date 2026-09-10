# PYTORCH_ALLOC_CONF

## 功能描述

通过此环境变量可配置TorchNPU缓存分配器行为，与`PYTORCH_NPU_ALLOC_CONF`功能相同。默认未配置。参数说明、默认值及支持范围请参见[PYTORCH_NPU_ALLOC_CONF](PYTORCH_NPU_ALLOC_CONF.md)。

该变量为与PyTorch统一命名对齐的通用入口，沿用PyTorch的同名环境变量。

## 配置示例

```bash
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
```

## 使用约束

- 从TorchNPU 26.2.0版本且PyTorch 2.10.0及以上版本开始支持，建议优先使用此变量。
- 与`PYTORCH_NPU_ALLOC_CONF`互斥，同时配置会报错并退出程序。

## 支持的型号

与[PYTORCH_NPU_ALLOC_CONF](PYTORCH_NPU_ALLOC_CONF.md#支持的型号)相同，各参数的产品约束也与该变量一致。
