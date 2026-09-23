# COMBINED\_ENABLE

## 功能描述

通过此环境变量可控制是否启用组合连续化优化。开启后，该功能旨在优化由多个 view 操作（如 reshape+slice、permute+select 等，最多支持 2 个组合操作）产生的非连续张量的连续化转换过程，通过推断 view 信息栈避免完整的内存拷贝。

- 配置为“0”：表示关闭此功能。
- 配置为“1”：表示开启。

默认配置为“0”。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export COMBINED_ENABLE=1
```

## 使用约束

- 必须在启动Python进程前配置该环境变量，进程运行过程中修改不会生效。
- 仅支持配置为“0”或“1”。
- 该环境变量仅在aclop路径的连续化优化中生效。走aclnn路径的算子不经过此优化，不受此环境变量影响。

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
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>
<!-- end id4 -->
