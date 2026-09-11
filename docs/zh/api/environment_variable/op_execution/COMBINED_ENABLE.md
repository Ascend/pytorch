# COMBINED\_ENABLE

## 功能描述

通过此环境变量可控制是否启用组合连续化优化。设置为“0”表示关闭此功能；设置为“1”表示开启，用于优化由多个view操作（如reshape+slice、permute+select等，最多2个组合操作）产生的非连续张量的连续化转换，通过推断view信息栈避免完整内存拷贝。

默认配置为“0”。

## 配置示例

```bash
export COMBINED_ENABLE=1
```

> [!NOTE]
>
> 此功能为`torch_npu`特有，PyTorch社区无直接对应变量。

## 使用约束

- 必须在启动Python进程前配置该环境变量，进程运行过程中修改不会生效。
- 仅支持配置为“0”或“1”。
- 该环境变量仅在aclop路径的连续化优化中生效。走aclnn路径的算子不经过此优化，不受此环境变量影响。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>
