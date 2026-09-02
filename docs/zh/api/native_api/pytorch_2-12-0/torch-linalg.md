# torch.linalg

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.12/linalg.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Matrix Properties](#matrix-properties)
- [Decompositions](#decompositions)
- [Solvers](#solvers)
- [Matrix Products](#matrix-products)
- [Experimental Functions](#experimental-functions)

</div>

<div style="display:none;">

## &#8203;torch.linalg

</div>

## Matrix Properties

### torch.linalg.norm

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.norm](https://pytorch.org/docs/2.12/generated/torch.linalg.norm.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

## Decompositions

### torch.linalg.cholesky

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.cholesky](https://pytorch.org/docs/2.12/generated/torch.linalg.cholesky.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

### torch.linalg.qr

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.qr](https://pytorch.org/docs/2.12/generated/torch.linalg.qr.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持fp32，fp64，complex64，complex128

</div>

## Solvers

### torch.linalg.lstsq

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.lstsq](https://pytorch.org/docs/2.12/generated/torch.linalg.lstsq.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： 可能回退至CPU执行

</div>

### torch.linalg.solve_triangular

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.solve_triangular](https://pytorch.org/docs/2.12/generated/torch.linalg.solve_triangular.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32，fp64，complex64，complex128

</div>

## Matrix Products

### torch.linalg.matmul

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.matmul](https://pytorch.org/docs/2.12/generated/torch.linalg.matmul.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：

- `input`仅支持fp16，fp32
- 输入最大支持6维

</div>

## Experimental Functions

### torch.linalg.ldl_factor

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.ldl_factor](https://pytorch.org/docs/2.12/generated/torch.linalg.ldl_factor.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>
