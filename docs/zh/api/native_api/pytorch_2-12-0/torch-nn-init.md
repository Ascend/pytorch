# torch.nn.init

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### torch.nn.init.calculate_gain

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.calculate_gain](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.calculate_gain)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.nn.init.uniform_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.uniform_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.uniform_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.init.normal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.normal_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.normal_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.init.constant_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.constant_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.constant_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.init.ones_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.ones_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.ones_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.init.zeros_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.zeros_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.zeros_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.init.eye_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.eye_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.eye_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.init.dirac_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.dirac_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.dirac_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.init.xavier_uniform_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.xavier_uniform_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.xavier_uniform_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.init.xavier_normal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.xavier_normal_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.xavier_normal_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.init.kaiming_uniform_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.kaiming_uniform_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.kaiming_uniform_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.init.kaiming_normal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.kaiming_normal_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.kaiming_normal_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.init.trunc_normal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.trunc_normal_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.trunc_normal_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.nn.init.orthogonal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.orthogonal_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.orthogonal_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.init.sparse_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.sparse_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.sparse_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持fp32

</div>
