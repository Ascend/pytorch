# PYTORCH_ALLOC_CONF

## 功能描述

PyTorch社区自2.10.0版本起正式发布`PYTORCH_ALLOC_CONF`环境变量。该变量是面向全部加速器后端（如CUDA、ROCm、XPU）的缓存分配器统一配置入口，不与具体设备绑定；`PYTORCH_CUDA_ALLOC_CONF`等设备专属变量名仅作为向后兼容的别名保留。

为保持与PyTorch社区命名一致，TorchNPU自26.2.0版本起引入该同名环境变量，用于配置NPU缓存分配器行为，功能与`PYTORCH_NPU_ALLOC_CONF`相同，建议优先使用此变量。默认未配置。参数说明、默认值及支持范围请参见[PYTORCH_NPU_ALLOC_CONF](PYTORCH_NPU_ALLOC_CONF.md)。

> [!NOTE]
>
> 与原生PyTorch同名变量相比，torch_npu在原生支持的参数基础上，还支持NPU专属参数（如`page_size`、`base_addr_aligned_kb`、`per_process_memory_fraction`等）。

## 配置示例

```bash
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
```

## 使用约束

- 该变量适用于TorchNPU 26.2.0且PyTorch 2.10.0及以上版本，建议优先使用此变量。
- 该变量与`PYTORCH_NPU_ALLOC_CONF`互斥，同时配置会报错并退出程序。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>
