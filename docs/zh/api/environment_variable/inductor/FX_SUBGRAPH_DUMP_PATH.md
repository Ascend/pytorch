# FX\_SUBGRAPH\_DUMP\_PATH

## 功能描述

通过此环境变量可配置FX子图的dump路径。设置后，在DVM模式（`TORCHINDUCTOR_NPU_BACKEND="dvm"`）下编译时会把FX子图导出到该路径（按`<路径>/<device索引>/<kernel名>`组织），用于子图划分与DVM编译问题的定位分析。

- 默认值未配置，不dump FX子图。
- 设置有效路径：将FX子图导出到指定目录。

PyTorch通过`torch._inductor.config`的debug输出控制，TorchNPU通过此环境变量提供相关配置。

## 配置示例

```bash
export FX_SUBGRAPH_DUMP_PATH=/tmp/fx_subgraph_dump
```

## 使用约束

- 需在导入`torch_npu`之前设置，并确保目标路径有写权限。
- 仅在DVM模式（`TORCHINDUCTOR_NPU_BACKEND="dvm"`）下生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
