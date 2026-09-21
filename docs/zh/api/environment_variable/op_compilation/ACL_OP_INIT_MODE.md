# ACL\_OP\_INIT\_MODE

## 功能描述

通过此环境变量可配置算子编译的初始化模式。

- 配置为“0”：aclops初始化模式，在NPU初始化阶段立即加载并初始化所有算子编译相关信息。
- 配置为“1”：aclops延迟初始化模式，算子编译相关信息在首次执行算子时延迟加载和初始化，可加速NPU初始化过程。
- 配置为“2”：禁用aclops，不加载算子编译相关信息。走aclop路径的算子将无法执行，执行时将抛出错误，提示检查`ACL_OP_INIT_MODE`配置，`jitCompile`等编译选项将不可用；走aclnn路径的算子不受影响，可正常执行。

默认值根据设备型号和CANN版本自动确定：

<!-- npu="950" id1 -->
- 仅支持aclnn的设备（如<term>Ascend 950DT</term>）默认配置为“2”，配置为其他值时将自动切换为“2”。
<!-- end id1 -->
<!-- npu="A3,910b" id2 -->
- <term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>（CANN >= 8.3.RC1），默认值为“1”。
<!-- end id2 -->
- 其他设备，默认值为“0”。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export ACL_OP_INIT_MODE=1
```

## 使用约束

- 必须在启动Python进程前配置该环境变量，进程运行过程中修改不会生效。
- 仅支持配置为“0”、“1”或“2”，其他取值将重置为默认值。

## 支持的型号

<!-- npu="910" id3 -->
- <term>Atlas 训练系列产品</term>
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品</term>
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 训练系列产品</term>
<!-- end id5 -->
<!-- npu="910b" id8 -->
- <term>Atlas 800I A2 推理产品</term>
<!-- end id8 -->
<!-- npu="310p" id6 -->
- <term>Atlas 推理系列产品</term>
<!-- end id6 -->
<!-- npu="950" id7 -->
- <term>Ascend 950DT</term>
<!-- end id7 -->
