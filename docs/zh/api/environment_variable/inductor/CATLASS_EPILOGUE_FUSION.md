# CATLASS\_EPILOGUE\_FUSION

## 功能描述

通过此环境变量可控制是否开启Catlass epilogue融合功能。开启后，Catlass模板库会将epilogue操作（如bias add、activation等）融合到matmul kernel中，减少额外的kernel launch和内存搬运。

- 默认值为`0`，关闭Catlass epilogue融合。
- 配置为`1`：开启Catlass epilogue融合。

该变量对应PyTorch的`CUTLASS_EPILOGUE_FUSION`。配置方式一致，默认值为`0`。

## 配置示例

开启Catlass epilogue融合：

```bash
export CATLASS_EPILOGUE_FUSION=1
```

## 使用约束

- 需在导入`torch_npu`之前设置。
- 需同时配置`TORCHINDUCTOR_NPU_CATLASS_DIR`（指定Catlass模板库路径）和`TORCHINDUCTOR_MAX_AUTOTUNE=1`时生效。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
