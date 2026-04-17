# TORCH\_TRANSFER\_TO\_NPU

## 功能描述

通过此环境变量可配置是否自动启用transfer\_to\_npu功能，将PyTorch的CUDA相关API自动替换为NPU对应API，方便用户从CUDA迁移到NPU。

- 配置为“1”时：启用transfer\_to\_npu功能，自动将torch.cuda相关接口替换为torch.npu对应接口，包括设备创建、张量操作、内存管理、流管理等。
- 配置为“0”或未配置时：不启用transfer\_to\_npu功能，用户需要手动使用torch.npu接口。
- 配置为其他值时：抛出ValueError异常，提示仅支持“0”或“1”。

此环境变量默认配置为“0”。

## 配置示例

启用transfer\_to\_npu功能：

```bash
export TORCH_TRANSFER_TO_NPU=1
```

禁用transfer\_to\_npu功能：

```bash
export TORCH_TRANSFER_TO_NPU=0
```

## 使用约束

- 此环境变量必须在导入torch之前设置，否则不生效。
- 更多transfer\_to\_npu相关约束参考《PyTorch 训练模型迁移调优指南》中的“[（推荐）自动迁移](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/recommended_auto_migration.md)”章节。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
