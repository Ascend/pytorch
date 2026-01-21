# 特征值检测

## 简介

特征值检测全称“训练阶段在线特征值检测“，主要针对长稳训练的使用场景，针对高比特跳变进行的检测。基于梯度做特征值的异常检测，识别精度问题。

检测原理和流程如下：

1.  在训练过程中采集需要检测的tensor，并计算特征值。
2.  异步线程接收特征值并进行异常检测。
3.  通过“**冷却抑制**”判断是否产生一次新的梯度异常，产生则打印一次“A grad-norm spike may happen”。

    冷却抑制：由于当前方案不做故障抑制，激活值上的异常梯度会影响多个参数，从而导致单张卡短时间内多次异常。因此，需要增加冷却机制，在第一次异常后一段时间内新的异常均认为是同一次异常，不计入“三振出局”校验。单次异常会打印“A grad-norm spike may happen”的Event日志，日志等级为INFO，默认不开启，可以通过设置“export TORCH\_NPU\_LOGS=silent”开启。

    “冷却抑制时间窗“通过配置项cooldown来控制，默认为5min。

4.  通过“**三振出局**”判断是否产生一次特征值异常，产生则打印一次“feature detection detects abnormal results”。

    三振出局：经过冷却抑制后，若确认为一次新的异常，则需要进一步进行“三振出局”检验，以判断当前三振出局时间窗内的异常总数是否达到了次数限制。如果判断达到了次数限制，则判定产生了特征值异常，打印“feature detection detects abnormal results”的Warning日志。

    “三振出局时间窗“通过配置项strikes\_window来控制，默认为480min。“三振出局异常次数“限制通过配置项strikes\_num配置，默认为3次。

5.  产生特征值异常时，在with\_checksum配置为true时，通知所有卡，同步启用“**checksum联动**”。

    checksum联动：当三振出局产生特征值异常时，如果配置开启了checksum联动功能，会同步在所有卡启用checksum联动检测。检测期间会存在额外的性能开销。具体原理是对torch.matmul和torch.Tensor.matmul的输入和输出进行checksum API的校验。在开启时间窗内如果存在校验为True的情况，则打印“The result of Matmul checksum is abnormal”的Warning日志。

    “单次checksum联动开启时间窗“复用“冷却抑制时间窗“，默认为5min。checksum在“checksum联动冷却时间窗“内仅允许开启一次，“checksum联动冷却时间窗“通过checksum\_cooldown来控制，默认为180min。

## 使用场景

检测模型训练过程中的梯度特征值是否存在异常。

## 使用指导

使用NPU\_ASD\_CONFIG环境变量开启或关闭特征值检测和checksum联动功能。

-   enable：可选配置为true或false，默认值为false。特征值检测是否开启的标志。
-   with\_checksum：可选配置为true或false，默认值为false。Checksum联动功能是否开启的标志。
-   cooldown：正整数，最小值为1，默认值为5，单位：分钟。冷却抑制时间窗，单次checksum联动开启时间窗复用冷却抑制时间窗，按需进行配置。
-   strikes\_num：正整数，最小值为1，默认值为3。三振出局异常次数限制，按需进行配置。
-   strikes\_window：正整数，最小值为1，默认值为480，单位：分钟。三振出局时间窗，按需进行配置。
-   checksum\_cooldown：正整数，最小值为1，默认值为180，单位：分钟。checksum联动冷却时间窗，按需进行配置。
-   upper\_thresh1：正整数，最小值为3，默认值为1000000。一级阈值，特征值超过绝对阈值会被认为是一次梯度异常。默认检测阈值无需配置，若需要修改阈值可通过配置环境变量修改。
-   upper\_thresh2：正整数，最小值为3，默认值为100。二级阈值，特征值超过二级阈值会被认为是一次疑似异常，不会更新到历史均值中。默认检测阈值无需配置，若需要修改阈值可通过此环境变量修改。
-   grad\_sample\_interval：正整数，最小值为1，默认值为3。梯度检测的间隔数，标记每多少个梯度中检测一个。配置越小检出率越高，但性能相对会劣化更严重，可能会超过2%。

此环境变量使用详情可参考《环境变量参考》中的“[NPU\_ASD\_CONFIG](../environment_variable_reference/NPU_ASD_CONFIG.md)”章节。

## 使用样例

```shell
export NPU_ASD_CONFIG=enable:true,with_checksum:true,cooldown:5,strikes_num:3,strikes_window:480,checksum_cooldown:180,upper_thresh1:1000000,upper_thresh2:100,grad_sample_interval:3
```

## 约束说明

-   当前仅能识别数据类型为**BF16**或**FP32**的模型训练过程中出现的梯度异常。
-   checksum联动仅支持**BF16**的数据类型。

