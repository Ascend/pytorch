# TORCHINDUCTOR_NPU_CATLASS_DIR（同社区TORCHINDUCTOR_CUTLASS_DIR）

## 功能描述

指定环境中Catlass库的路径，该环境变量与社区CUTLASS库路径保持一致，社区环境变量为TORCHINDUCTOR_CUTLASS_DIR。

若路径配置错误，会有WARNING信息提示，并跳过引入Catlass后端的功能。

未设置时默认使用torch安装目录上一级的third_party/catlass（即`<torch安装目录>/../third_party/catlass`）；若环境中已安装catlass_cppgen，则优先使用其所在路径。

## 配置示例

配置Catlass库路径为/path/to/catlass/dir：

```shell
export TORCHINDUCTOR_NPU_CATLASS_DIR="/path/to/catlass/dir"
```

## 使用约束

无

## 支持的型号

<term>Ascend 950PR&950DT系列产品</term>
