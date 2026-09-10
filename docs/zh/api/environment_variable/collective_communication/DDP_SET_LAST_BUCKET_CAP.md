# DDP\_SET\_LAST\_BUCKET\_CAP

## 功能描述

通过此环境变量可控制DDP（Distributed Data Parallel）的bucket重建顺序。当设为"1"时，较小的first bucket cap会被分配到最后一个bucket，而不是第一个bucket。

- 精确等于字符串"1"：启用，将较小的first bucket cap落到最后一个bucket。
- 其他值：不启用，保持默认的bucket顺序。

默认值：未设置。

该变量沿用PyTorch的同名环境变量，配置方式一致。

## 配置示例

```bash
export DDP_SET_LAST_BUCKET_CAP=1
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
