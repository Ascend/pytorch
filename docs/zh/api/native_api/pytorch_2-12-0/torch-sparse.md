# torch.sparse

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.12/sparse.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Supported operations](#supported-operations)

</div>

<div style="display:none;">

## &#8203;torch.sparse

</div>

## Supported operations

### <code><i>class</i></code> torch.Tensor

<div style="margin-left: 2em">

> <font size="3">is_sparse_csr()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_sparse_csr](https://pytorch.org/docs/2.12/generated/torch.Tensor.is_sparse_csr.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">to_sparse_coo()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.to_sparse_coo](https://pytorch.org/docs/2.12/generated/torch.Tensor.to_sparse_coo.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">coalesce()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.coalesce](https://pytorch.org/docs/2.12/generated/torch.Tensor.coalesce.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">sparse_resize_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sparse_resize_](https://pytorch.org/docs/2.12/generated/torch.Tensor.sparse_resize_.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">sparse_resize_and_clear_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sparse_resize_and_clear_](https://pytorch.org/docs/2.12/generated/torch.Tensor.sparse_resize_and_clear_.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_coalesced()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_coalesced](https://pytorch.org/docs/2.12/generated/torch.Tensor.is_coalesced.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">crow_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.crow_indices](https://pytorch.org/docs/2.12/generated/torch.Tensor.crow_indices.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">col_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.col_indices](https://pytorch.org/docs/2.12/generated/torch.Tensor.col_indices.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">row_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.row_indices](https://pytorch.org/docs/2.12/generated/torch.Tensor.row_indices.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">ccol_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ccol_indices](https://pytorch.org/docs/2.12/generated/torch.Tensor.ccol_indices.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.sparse.check_sparse_tensor_invariants

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.check_sparse_tensor_invariants](https://pytorch.org/docs/2.12/generated/torch.sparse.check_sparse_tensor_invariants.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">disable()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.check_sparse_tensor_invariants.disable](https://pytorch.org/docs/2.12/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants.disable)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">enable()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.check_sparse_tensor_invariants.enable](https://pytorch.org/docs/2.12/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants.enable)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_enabled()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.check_sparse_tensor_invariants.is_enabled](https://pytorch.org/docs/2.12/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants.is_enabled)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### torch.sparse_compressed_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_compressed_tensor](https://pytorch.org/docs/2.12/generated/torch.sparse_compressed_tensor.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.sparse.as_sparse_gradcheck

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.as_sparse_gradcheck](https://pytorch.org/docs/2.12/generated/torch.sparse.as_sparse_gradcheck.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>
