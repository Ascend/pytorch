# torch.utils.deterministic

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### torch.utils.deterministic.fill_uninitialized_memory

<div style="margin-left: 2em">

**原生文档**：[torch.utils.deterministic.fill_uninitialized_memory](https://pytorch.org/docs/2.12/deterministic.html#torch.utils.deterministic.fill_uninitialized_memory)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**：`torch.utils.deterministic.fill_uninitialized_memory`默认值为True。设置`torch.use_deterministic_algorithms`开启确定性后，PyTorch行为是填充未初始化内存，而TorchNPU默认不填充。如需填充生效，需手动将`torch.utils.deterministic.fill_uninitialized_memory`设置为True

</div>
