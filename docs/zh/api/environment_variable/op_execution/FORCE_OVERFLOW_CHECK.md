# FORCE\_OVERFLOW\_CHECK

## 功能描述

通过此环境变量可在非饱和模式（INF\_NAN模式）下开启溢出检测开关，用于训练过程中Inf/NaN问题的异步定位，不改变浮点计算模式。环境变量默认值为“0”。

- 环境变量值为“0”时：代表不开启溢出检测开关，行为与未配置时一致。
- 环境变量值为“1”时：代表开启溢出检测开关。

当环境变量开启时，溢出检测接口（[get\_npu\_overflow\_flag](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/（beta）torch_npu-npu-get_npu_overflow_flag.md)、[npu\_check\_overflow](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/（beta）torch_npu-npu-utils-npu_check_overflow.md)、[clear\_npu\_overflow\_flag](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/（beta）torch_npu-npu-clear_npu_overflow_flag.md)）在INF\_NAN模式下可用，可通过溢出状态标志判断数值是否溢出，无需将梯度搬运至Host侧判断是否为Inf/NaN，避免引入同步等待，适合对执行时序敏感的溢出定位场景。

<!-- npu="A3,910b" id5 -->
以下为<term>Atlas A2训练系列产品</term>/<term>Atlas A3训练系列产品</term>使用get\_npu\_overflow\_flag的示例（其他溢出检测接口使用方式与此类似）：

```python
import torch
import torch_npu
a = torch.tensor([65504.], dtype=torch.float16).npu()
b = a + a  # fp16溢出，INF_NAN模式下结果为inf
torch_npu.npu.synchronize()
print(torch_npu.npu.utils.get_npu_overflow_flag())  # True，通过溢出状态标志检出
```
<!-- end id5 -->

> [!NOTE]  
>
> <!-- npu="950" id4 -->
> - <term>Ascend 950DT系列产品</term>不支持使用`get_npu_overflow_flag`等接口查询溢出状态；当<term>Ascend 950DT系列产品</term>开启该环境变量开关后，需通过plog中的`aclrtSetStreamOverflowSwitch`相关日志（需配置`ASCEND_GLOBAL_LOG_LEVEL=1`）的方式确认溢出检测开关已生效。
> <!-- end id4 -->
> - 开启溢出检测开关后，算子执行性能可能受到影响，建议仅在溢出/NaN问题定位场景下配置。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export FORCE_OVERFLOW_CHECK=1
```

## 使用约束

- 需CANN版本不低于9.1.0，版本不满足时打印WARNING日志并忽略该环境变量。
- 仅支持配置为“0”或“1”，其他取值视为未配置。
- 需在进程启动前配置，进程运行过程中修改不会生效。

## 支持的型号

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id3 -->
