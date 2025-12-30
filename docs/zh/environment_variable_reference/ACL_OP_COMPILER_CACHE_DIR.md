# ACL\_OP\_COMPILER\_CACHE\_DIR

## 功能描述

通过此环境变量可配置算子编译磁盘缓存的目录。

优先级：ACL\_OP\_COMPILER\_CACHE\_DIR \> ASCEND\_CACHE\_PATH   \> 默认路径（$HOME/atc\_data）

如果设置了该环境变量，则按照该环境变量指定路径落盘算子编译缓存；未设置则使用ASCEND\_CACHE\_PATH指定路径；若未设置该环境变量且未使用ASCEND\_CACHE\_PATH指定路径，则使用默认路径（$HOME/atc\_data）。

## 配置示例

```
export ACL_OP_COMPILER_CACHE_DIR=/home/cache
```

## 使用约束

-   该环境变量仅在单算子模式下可以使用，图模式不支持该环境变量。
-   该环境变量需要与ACL\_OP\_COMPILER\_CACHE\_MODE配合使用。
-   如果同时设置了环境变量和torch\_npu\_option，则以代码中的torch\_npu\_option方式为优先。
-   如果设置了ACL\_OP\_DEBUG\_LEVEL编译选项，则只有编译选项值为0或3才会启用编译缓存功能，其它取值禁用编译缓存功能。ACL\_OP\_DEBUG\_LEVEL编译选项具体可参考《CANN  应用开发接口》的“[aclCompileOpt](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/appdevgapi/aclcppdevg_03_1371.html)”章节。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 800I A2 推理产品</term>

