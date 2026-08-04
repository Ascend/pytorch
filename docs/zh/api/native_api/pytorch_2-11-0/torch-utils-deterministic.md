# torch.utils.deterministic

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### torch.utils.deterministic.fill_uninitialized_memory

<div style="margin-left: 2em">

**原生文档**：[torch.utils.deterministic.fill_uninitialized_memory](https://pytorch.org/docs/2.11/deterministic.html#torch.utils.deterministic.fill_uninitialized_memory)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**: torch.utils.deterministic.fill_uninitialized_memory默认值True，设置torch.use_deterministic_algorithms开启确定性后社区行为是填充，NPU上默认不填充，需要手动设置torch.utils.deterministic.fill_uninitialized_memory后NPU上填充才能生效

</div>
