# ACL\_OP\_COMPILER\_CACHE\_MODE

## 功能描述

通过此环境变量可配置算子编译磁盘缓存模式。

-   配置为`enable`：启用算子编译缓存。启用后可以避免针对相同编译参数及算子参数的算子重复编译，从而提升编译速度。
-   配置为`disable`：禁用算子编译缓存。
-   配置为`force`：强制刷新缓存。当用户的Python或者依赖库等发生变化时，需要指定为`force`用于清理已有的缓存。

默认配置为`enable`。

## 配置示例

```
export ACL_OP_COMPILER_CACHE_MODE=enable
```

## 使用约束

-   该环境变量仅在单算子模式下可以使用，图模式不支持该环境变量。
-   开启算子编译缓存的场景下，可以通过[ACL\_OP\_COMPILER\_CACHE\_DIR](ACL_OP_COMPILER_CACHE_DIR.md)配置算子编译缓存文件存储路径。
-   如果同时通过此环境变量和torch\_npu\_option方式指定算子编译磁盘缓存模式，以代码中的torch\_npu\_option方式为优先。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 800I A2 推理产品</term>

