# TORCH\_CACHING\_PRECOMPILE

## 功能描述

通过此环境变量可开启自动缓存预编译（automatic caching precompile）实验性功能。开启后，`torch.compile`会在编译过程中自动将Dynamo缓存条目与后端编译产物序列化到全局Precompile Cache中，并在后续运行中自动加载复用，从而跳过重复编译，加速后续编译过程。

- 默认值为`0`，关闭自动缓存预编译。
- 配置为`1`：开启自动缓存预编译。

开启后，系统会进行以下处理：

- **编译时自动加载**：`torch.compile`调用时自动从DynamoCache查找并加载已有的编译缓存。
- **编译后自动保存**：编译完成后自动将Dynamo缓存条目与后端产物（如AOTAutograd cache）写入DynamoCache。
- **Guard过滤**：自动丢弃不可序列化的guard（如`ID_MATCH`、`CLOSURE_MATCH`、`WEAKREF_ALIVE`、`DICT_VERSION`），以确保缓存可跨进程复用。
- **与`torch.compiler.save_cache_artifacts()`配合**：调用`save_cache_artifacts()`时会将`PrecompileContext`中的缓存条目一并保存。

> [!NOTE]
>
> - 此功能为实验性功能，适用于需要多次编译相同模型的场景（如训练多轮次），可显著减少后续编译时间。
> - 开启后会丢弃部分guard，可能导致缓存命中条件放宽，使用时需关注编译结果的正确性。
> - 建议在与`torch.compiler.save_cache_artifacts()`/`load_cache_artifacts()`配合使用时开启。

该变量沿用PyTorch的同名环境变量，配置方式一致。

## 配置示例

开启自动缓存预编译：

```bash
export TORCH_CACHING_PRECOMPILE=1
```

## 使用约束

需在进程启动前配置，进程运行过程中修改不会生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
