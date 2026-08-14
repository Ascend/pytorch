# torch.sparse

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.11/sparse.html)。

## 目录

- [Supported operations](#supported-operations)

## Supported operations

### <code><i>class</i></code> torch.sparse.Tensor

<div style="margin-left: 2em">

> <font size="3">is_sparse_csr()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">to_sparse_coo()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">coalesce()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">sparse_resize_()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">sparse_resize_and_clear_()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_coalesced()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">crow_indices()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">col_indices()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">row_indices()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">ccol_indices()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### torch.sparse.check_sparse_tensor_invariants.disable

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.sparse.check_sparse_tensor_invariants.enable

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.sparse.check_sparse_tensor_invariants.is_enabled

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.sparse_compressed_tensor

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.sparse.check_sparse_tensor_invariants

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.sparse.as_sparse_gradcheck

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>
