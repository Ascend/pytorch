# torch.sparse

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.9/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.9/sparse.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Supported operations](#supported-operations)

</div>

<div style="display:none;">

## &#8203;torch.sparse

</div>

### torch.sparse.sum

<div style="margin-left: 2em">

**原生文档**：[torch.sparse.sum](https://docs.pytorch.org/docs/2.9/generated/torch.sparse.sum.html#torch.sparse.sum)

**产品支持情况**：

<!-- npu="910b" id59 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="A3" id60 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id60 -->
<!-- npu="950" id61 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id61 -->

</div>

## Supported operations

### <code><i>class</i></code> torch.Tensor

<div style="margin-left: 2em">

<!-- npu="910b,A3" id45 -->
> <font size="3">is_sparse_csr()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_sparse_csr](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_sparse_csr.html)

**产品支持情况**：
<!-- end id45 -->

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->

<!-- npu="910b,A3" id46 -->
</div>
<!-- end id46 -->

<!-- npu="910b,A3" id47 -->
> <font size="3">to_sparse_coo()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.to_sparse_coo](https://pytorch.org/docs/2.9/generated/torch.Tensor.to_sparse_coo.html)

**产品支持情况**：
<!-- end id47 -->

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->

<!-- npu="910b,A3" id48 -->
</div>
<!-- end id48 -->

> <font size="3">coalesce()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.coalesce](https://pytorch.org/docs/2.9/generated/torch.Tensor.coalesce.html)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id9 -->

</div>

> <font size="3">sparse_resize_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sparse_resize_](https://pytorch.org/docs/2.9/generated/torch.Tensor.sparse_resize_.html)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id12 -->

</div>

> <font size="3">sparse_resize_and_clear_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sparse_resize_and_clear_](https://pytorch.org/docs/2.9/generated/torch.Tensor.sparse_resize_and_clear_.html)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id15 -->

</div>

> <font size="3">is_coalesced()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_coalesced](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_coalesced.html)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id18 -->

</div>

<!-- npu="910b,A3" id49 -->
> <font size="3">crow_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.crow_indices](https://pytorch.org/docs/2.9/generated/torch.Tensor.crow_indices.html)

**产品支持情况**：
<!-- end id49 -->

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->

<!-- npu="910b,A3" id50 -->
</div>
<!-- end id50 -->

<!-- npu="910b,A3" id51 -->
> <font size="3">col_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.col_indices](https://pytorch.org/docs/2.9/generated/torch.Tensor.col_indices.html)

**产品支持情况**：
<!-- end id51 -->

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->

<!-- npu="910b,A3" id52 -->
</div>
<!-- end id52 -->

<!-- npu="910b,A3" id53 -->
> <font size="3">row_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.row_indices](https://pytorch.org/docs/2.9/generated/torch.Tensor.row_indices.html)

**产品支持情况**：
<!-- end id53 -->

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->

<!-- npu="910b,A3" id54 -->
</div>
<!-- end id54 -->

<!-- npu="910b,A3" id55 -->
> <font size="3">ccol_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ccol_indices](https://pytorch.org/docs/2.9/generated/torch.Tensor.ccol_indices.html)

**产品支持情况**：
<!-- end id55 -->

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->

<!-- npu="910b,A3" id56 -->
</div>
<!-- end id56 -->

</div>

<!-- npu="910b,A3" id57 -->
### torch.sparse_compressed_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_compressed_tensor](https://pytorch.org/docs/2.9/generated/torch.sparse_compressed_tensor.html)

**产品支持情况**：
<!-- end id57 -->

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->

<!-- npu="910b,A3" id58 -->
</div>
<!-- end id58 -->
