# torch.testing

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### torch.testing.assert_close

<div style="margin-left: 2em">

**原生文档**：[torch.testing.assert_close](https://pytorch.org/docs/2.11/testing.html#torch.testing.assert_close)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： 支持fp32

</div>

### torch.testing.make_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.testing.make_tensor](https://pytorch.org/docs/2.11/testing.html#torch.testing.make_tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： 支持fp32，bool

</div>

### torch.testing.assert_allclose

<div style="margin-left: 2em">

**原生文档**：[torch.testing.assert_allclose](https://pytorch.org/docs/2.11/testing.html#torch.testing.assert_allclose)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： 支持fp32

</div>
