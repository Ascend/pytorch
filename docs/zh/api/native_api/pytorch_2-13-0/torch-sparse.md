# torch.sparse

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)
- [Sparse Compressed Tensors](#sparse-compressed-tensors)
- [Sparse COO tensors](#sparse-coo-tensors)
- [Supported operations](#supported-operations)

## base API

### _`class`_ torch.sparse.Tensor

<div style="margin-left: 2em">

> <font size="3">is_sparse_csr()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_sparse_csr](https://pytorch.org/docs/2.13/generated/torch.Tensor.is_sparse_csr.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">to_sparse_coo()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.to_sparse_coo](https://pytorch.org/docs/2.13/generated/torch.Tensor.to_sparse_coo.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">coalesce()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.coalesce](https://pytorch.org/docs/2.13/generated/torch.Tensor.coalesce.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">sparse_resize_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sparse_resize_](https://pytorch.org/docs/2.13/generated/torch.Tensor.sparse_resize_.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">sparse_resize_and_clear_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sparse_resize_and_clear_](https://pytorch.org/docs/2.13/generated/torch.Tensor.sparse_resize_and_clear_.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">is_coalesced()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_coalesced](https://pytorch.org/docs/2.13/generated/torch.Tensor.is_coalesced.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">crow_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.crow_indices](https://pytorch.org/docs/2.13/generated/torch.Tensor.crow_indices.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">col_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.col_indices](https://pytorch.org/docs/2.13/generated/torch.Tensor.col_indices.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">row_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.row_indices](https://pytorch.org/docs/2.13/generated/torch.Tensor.row_indices.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">ccol_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ccol_indices](https://pytorch.org/docs/2.13/generated/torch.Tensor.ccol_indices.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### torch.sparse.check_sparse_tensor_invariants.disable

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.check_sparse_tensor_invariants.disable](https://pytorch.org/docs/2.13/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants.disable)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.sparse.check_sparse_tensor_invariants.enable

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.check_sparse_tensor_invariants.enable](https://pytorch.org/docs/2.13/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants.enable)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.sparse.check_sparse_tensor_invariants.is_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.check_sparse_tensor_invariants.is_enabled](https://pytorch.org/docs/2.13/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants.is_enabled)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

## Sparse Compressed Tensors

### torch.sparse_compressed_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_compressed_tensor](https://pytorch.org/docs/2.13/generated/torch.sparse_compressed_tensor.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

## Sparse COO tensors

### torch.sparse.check_sparse_tensor_invariants

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.check_sparse_tensor_invariants](https://pytorch.org/docs/2.13/generated/torch.sparse.check_sparse_tensor_invariants.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

## Supported operations

### torch.sparse.as_sparse_gradcheck

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.as_sparse_gradcheck](https://pytorch.org/docs/2.13/generated/torch.sparse.as_sparse_gradcheck.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>
