# FORCE\_OVERFLOW\_CHECK

## 功能描述

通过此环境变量可在非饱和模式（INF_NAN模式）下开启溢出检测开关，用于训练过程中Inf/NaN问题的异步定位，不改变浮点计算模式。环境变量默认值为“0”。

- 环境变量值为“0”时：代表不开启溢出检测开关，行为与未配置时一致。
- 环境变量值为“1”时：代表开启溢出检测开关，溢出检测接口（[get\_npu\_overflow\_flag](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/（beta）torch_npu-npu-get_npu_overflow_flag.md)、[npu\_check\_overflow](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/（beta）torch_npu-npu-utils-npu_check_overflow.md)、[clear\_npu\_overflow\_flag](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/（beta）torch_npu-npu-clear_npu_overflow_flag.md)）在INF\_NAN模式下可用：通过读取溢出状态标志判断是否发生过数值溢出，而非将梯度搬运至Host侧判断是否为Inf/NaN，因此不会引入同步等待，适合对执行时序敏感的溢出定位场景。

针对<term>Atlas 训练系列产品</term>/<term>Atlas 推理系列产品</term>/<term>Atlas 200I/500 A2 推理产品</term>，仅支持饱和模式，溢出检测开关默认开启，该环境变量不生效。

> [!NOTICE]  
>
> 开启溢出检测开关后，算子执行性能可能受到影响，建议仅在溢出/NaN问题定位场景下配置。

在INF\_NAN模式下配置FORCE\_OVERFLOW\_CHECK=1后，可通过溢出状态标志检测数值溢出：

```python
import torch
import torch_npu
a = torch.tensor([65504.], dtype=torch.float16).npu()
b = a + a  # fp16溢出，INF_NAN模式下结果为inf
torch_npu.npu.synchronize()
print(torch_npu.npu.utils.get_npu_overflow_flag())  # True，通过溢出状态标志检出
```

## 配置示例

```bash
export FORCE_OVERFLOW_CHECK=1
```

## 使用约束

- 需CANN版本不低于9.1.0，版本不满足时该环境变量不生效。
- 仅支持配置为“0”或“1”，其他取值视为未配置。
- 需在进程启动前配置，进程运行过程中修改不会生效。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
