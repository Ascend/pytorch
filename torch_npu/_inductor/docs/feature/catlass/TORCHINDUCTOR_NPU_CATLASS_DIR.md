## TORCHINDUCTOR_NPU_CATLASS_DIR （同社区TORCHINDUCTOR_CUTLASS_DIR）

## 功能描述

环境中catlass库的路径，该环境变量与社区CUTLASS库路径保持一致，社区环境变量为TORCHINDUCTOR_CUTLASS_DIR。

若路径配置错误，会有WARNING信息提示，并跳过尝试引入catlass后端的功能。

默认配置为TORCHINDUCTOR_NPU_CATLASS_DIR=""。

## 配置示例

配置catlass库路径为/path/to/catlass/dir。

```shell
export TORCHINDUCTOR_NPU_CATLASS_DIR="/path/to/catlass/dir"
```

## 使用约束

无

## 支持的型号

-   <term>Atlas A5 系列产品</term>