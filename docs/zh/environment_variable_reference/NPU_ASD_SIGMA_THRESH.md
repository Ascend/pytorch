# NPU\_ASD\_SIGMA\_THRESH

## 功能描述

通过此环境变量可配置特征值检测功能的相对阈值，格式为整型数据对，最小取值为3。

算法会检测特征值跳变幅度，其中第一个元素控制一级阈值，特征值跳变幅度超过一级阈值时，会终止训练并上报告警；第二个元素控制二级阈值，特征值跳变幅度超过二级阈值且`ASCEND_GLOBAL_LOG_LEVEL`设置为“0”、“1”或“2”时，会打印Warning级别日志预警。减小阈值可以检出波动更小的异常数据，增加检出率，增大阈值与之相反。

特征值检测功能具体参见《PyTorch 训练模型迁移调优指南》的“[特征值检测](https://www.hiascend.com/document/detail/zh/Pytorch/720/ptmoddevg/trainingmigrguide/PT_LMTMOG_0024.html)”章节。

默认阈值为100000,5000。

> [!NOTE]  
> 出厂默认阈值即为最优值，不推荐客户修改。若发生如下情况可根据实际场景调整阈值，并注意相关影响。
> -   需要调大阈值场景：若发生告警，并且确认此次数值波动为正常，不影响训练，则调大阈值。
>     -   若val超过NPU\_ASD\_UPPER\_THRESH导致告警，则需根据val值调大阈值NPU\_ASD\_UPPER\_THRESH（推荐为val\*2）；
>     -   若跳变幅度超过NPU\_ASD\_SIGMA\_THRESH触发告警，则需根据\(val-pre\_val\)和\(max-min\)的比值调大阈值NPU\_ASD\_SIGMA\_THRESH（推荐为\(val-pre\_val\)/\(max-min\)\*2 ）。
> 
>     相关影响：调大阈值会导致检出率有所减低，但误检率也会降低。
> -   需要调小阈值场景：若频繁出现loss spike/grad norm spike影响训练，重新拉起依然有spike，但无告警，则按照一定比例（如10）逐渐调小阈值。  
>     相关影响：调小阈值能够提高检出率，但也容易引发误检。

## 配置示例

```
export NPU_ASD_SIGMA_THRESH=100000,5000
```

## 使用约束

-   此环境变量不支持在PyTorch图模式（TorchAir）场景下使用。

-   此环境变量适用于Ascend Extension for PyTorch 7.0.0及之前版本。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 推理系列产品</term>

