# NSLB\_MAX\_RECORD\_NUM

## 功能描述

通过此环境变量可控制每个PG的NSLB（Non-Stop Load Balancing）最大采样记录数量。

默认值：1000。达到上限后，会写入end文件并停止采样。

> [!NOTE]
>
> - 仅当`NSLB_CP`设置为非空路径时，采样记录才会写入文件。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export NSLB_MAX_RECORD_NUM=5000
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
