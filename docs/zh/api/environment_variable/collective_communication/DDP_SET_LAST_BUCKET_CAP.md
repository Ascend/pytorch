# DDP\_SET\_LAST\_BUCKET\_CAP

## 功能描述

通过此环境变量可控制DDP（Distributed Data Parallel）的bucket重建顺序。当设为开启时，较小的`first bucket cap`会被分配到最后一个bucket，而不是第一个bucket。该环境变量默认未配置。

- 配置为“1”：启用，将较小的`first bucket cap`落到最后一个bucket。
- 其他值或未配置：不启用，保持默认的bucket顺序。

该变量沿用PyTorch的同名环境变量，配置方式一致。

## 配置示例

```bash
export DDP_SET_LAST_BUCKET_CAP=1
```

## 使用约束

无

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
