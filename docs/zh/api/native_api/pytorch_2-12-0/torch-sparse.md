# torch.sparse

> [!NOTE]
>
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

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

**原生文档**：[Tensor.is_sparse_csr](https://pytorch.org/docs/2.12/generated/torch.Tensor.is_sparse_csr.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">to_sparse_coo()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.to_sparse_coo](https://pytorch.org/docs/2.12/generated/torch.Tensor.to_sparse_coo.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">coalesce()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.coalesce](https://pytorch.org/docs/2.12/generated/torch.Tensor.coalesce.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">sparse_resize_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sparse_resize_](https://pytorch.org/docs/2.12/generated/torch.Tensor.sparse_resize_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">sparse_resize_and_clear_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sparse_resize_and_clear_](https://pytorch.org/docs/2.12/generated/torch.Tensor.sparse_resize_and_clear_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">is_coalesced()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_coalesced](https://pytorch.org/docs/2.12/generated/torch.Tensor.is_coalesced.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">crow_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.crow_indices](https://pytorch.org/docs/2.12/generated/torch.Tensor.crow_indices.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">col_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.col_indices](https://pytorch.org/docs/2.12/generated/torch.Tensor.col_indices.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">row_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.row_indices](https://pytorch.org/docs/2.12/generated/torch.Tensor.row_indices.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">ccol_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ccol_indices](https://pytorch.org/docs/2.12/generated/torch.Tensor.ccol_indices.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### torch.sparse.check_sparse_tensor_invariants.disable

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.check_sparse_tensor_invariants.disable](https://pytorch.org/docs/2.12/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants.disable)

**是否支持**：否

</div>

### torch.sparse.check_sparse_tensor_invariants.enable

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.check_sparse_tensor_invariants.enable](https://pytorch.org/docs/2.12/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants.enable)

**是否支持**：否

</div>

### torch.sparse.check_sparse_tensor_invariants.is_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.check_sparse_tensor_invariants.is_enabled](https://pytorch.org/docs/2.12/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants.is_enabled)

**是否支持**：否

</div>

## Sparse Compressed Tensors

### torch.sparse_compressed_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_compressed_tensor](https://pytorch.org/docs/2.12/generated/torch.sparse_compressed_tensor.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Sparse COO tensors

### torch.sparse.check_sparse_tensor_invariants

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.check_sparse_tensor_invariants](https://pytorch.org/docs/2.12/generated/torch.sparse.check_sparse_tensor_invariants.html)

**是否支持**：否

</div>

## Supported operations

### torch.sparse.as_sparse_gradcheck

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.as_sparse_gradcheck](https://pytorch.org/docs/2.12/generated/torch.sparse.as_sparse_gradcheck.html)

**是否支持**：否

</div>
