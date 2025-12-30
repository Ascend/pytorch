# NPU\_ASD\_CONFIG

## 功能描述

通过此环境变量可控制是否开启Ascend Extension for PyTorch的特征值检测功能。特征值检测功能具体参见《PyTorch 框架特性指南》中的“[特征值检测](https://www.hiascend.com/document/detail/zh/Pytorch/720/ptmoddevg/Frameworkfeatures/featuresguide_00013.html)”章节。

该环境变量支持以下可选参数：

-   enable：可选配置为true或false，默认值为false。特征值检测是否开启的标志。
-   with\_checksum：可选配置为true或false，默认值为false。checksum联动功能是否开启的标志。
-   cooldown：正整数，最小值为1，默认值为5，单位：分钟。冷却抑制时间窗，checksum联动单次开启时间窗，按需进行配置。
-   strikes\_num：正整数，最小值为1，默认值为3。三振出局异常次数限制，按需进行配置。
-   strikes\_window：正整数，最小值为1，默认值为480，单位：分钟。三振出局检测时间窗，按需进行配置。
-   checksum\_cooldown：正整数，最小值为1，默认值为180，单位：分钟。checksum联动冷却时间窗，按需进行配置。
-   upper\_thresh1：正整数，最小值为3，默认值为1000000。一级阈值，特征值超过绝对阈值会被认为是一次梯度异常。默认检测阈值无需配置，若需要修改阈值可通过配置环境变量修改。
-   upper\_thresh2：正整数，最小值为3，默认值为100。二级阈值，特征值超过二级阈值会被认为是一次疑似异常，不会更新到历史均值中。默认检测阈值无需配置，若需要修改阈值可通过此环境变量修改。
-   grad\_sample\_interval：正整数，最小值为1，默认值为3。梯度检测的间隔数，标记多少个梯度中检测一个。配置越小检出率越高，但性能相对会劣化更严重，性能下降可能会超过2%。

## 配置示例

```
export NPU_ASD_CONFIG=enable:true,with_checksum:true,cooldown:5,strikes_num:3,strikes_window:480,checksum_cooldown:180,upper_thresh1:1000000,upper_thresh2:100,grad_sample_interval:3
```

## 使用约束

-   此环境变量不支持在PyTorch图模式（TorchAir）场景下使用。
-   特征值检测需要计算激活值梯度的统计值，会产生额外的显存占用，最多可能存在1.5G的额外显存消耗，用户显存紧张的情况下可能导致OOM（Out of Memory，内存不足）。
-   此环境变量适用于Ascend Extension for PyTorch 7.1.0及之后版本。Ascend Extension for PyTorch 7.0.0及之前，可使用[NPU\_ASD\_ENABLE](NPU_ASD_ENABLE.md)开启特征值检测，具体操作可参考Ascend Extension for PyTorch对应版本资料。
-   当前仅能识别数据类型为**BF16**或**FP32**的模型训练过程中出现的梯度异常。
-   checksum联动仅支持**BF16**的数据类型。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>

