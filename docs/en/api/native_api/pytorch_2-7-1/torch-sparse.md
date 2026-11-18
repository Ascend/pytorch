# torch.sparse

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/sparse.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Supported operations](#supported-operations)

</div>

<div style="display:none;">

## &#8203;torch.sparse

</div>

### torch.sparse.sum

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse.sum](https://docs.pytorch.org/docs/2.7/generated/torch.sparse.sum.html#torch.sparse.sum)

**Support Status**:

<!-- npu="910b" id59 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id59 -->
<!-- npu="A3" id60 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id60 -->
<!-- npu="950" id61 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id61 -->

</div>

## Supported operations

### <code><i>class</i></code> torch.Tensor

<div style="margin-left: 2em">

<!-- npu="910b,A3" id45 -->
> <font size="3">is_sparse_csr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_sparse_csr](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_sparse_csr.html)

**Support Status**:
<!-- end id45 -->

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->

<!-- npu="910b,A3" id46 -->
</div>
<!-- end id46 -->

<!-- npu="910b,A3" id47 -->
> <font size="3">to_sparse_coo()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.to_sparse_coo](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_sparse_coo.html)

**Support Status**:
<!-- end id47 -->

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->

<!-- npu="910b,A3" id48 -->
</div>
<!-- end id48 -->

> <font size="3">coalesce()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.coalesce](https://pytorch.org/docs/2.7/generated/torch.Tensor.coalesce.html)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

</div>

> <font size="3">sparse_resize_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sparse_resize_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sparse_resize_.html)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

</div>

> <font size="3">sparse_resize_and_clear_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sparse_resize_and_clear_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sparse_resize_and_clear_.html)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id15 -->

</div>

> <font size="3">is_coalesced()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_coalesced](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_coalesced.html)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id18 -->

</div>

<!-- npu="910b,A3" id49 -->
> <font size="3">crow_indices()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.crow_indices](https://pytorch.org/docs/2.7/generated/torch.Tensor.crow_indices.html)

**Support Status**:
<!-- end id49 -->

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->

<!-- npu="910b,A3" id50 -->
</div>
<!-- end id50 -->

<!-- npu="910b,A3" id51 -->
> <font size="3">col_indices()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.col_indices](https://pytorch.org/docs/2.7/generated/torch.Tensor.col_indices.html)

**Support Status**:
<!-- end id51 -->

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->

<!-- npu="910b,A3" id52 -->
</div>
<!-- end id52 -->

<!-- npu="910b,A3" id53 -->
> <font size="3">row_indices()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.row_indices](https://pytorch.org/docs/2.7/generated/torch.Tensor.row_indices.html)

**Support Status**:
<!-- end id53 -->

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->

<!-- npu="910b,A3" id54 -->
</div>
<!-- end id54 -->

<!-- npu="910b,A3" id55 -->
> <font size="3">ccol_indices()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ccol_indices](https://pytorch.org/docs/2.7/generated/torch.Tensor.ccol_indices.html)

**Support Status**:
<!-- end id55 -->

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->

<!-- npu="910b,A3" id56 -->
</div>
<!-- end id56 -->

</div>

### torch.sparse.mm

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse.mm](https://docs.pytorch.org/docs/2.7/generated/torch.sparse.mm.html#torch.sparse.mm)

**Support Status**:

<!-- npu="910b" id62 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id62 -->
<!-- npu="A3" id63 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id63 -->
<!-- npu="950" id64 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id64 -->

</div>

<!-- npu="910b,A3" id65 -->
### torch.sparse_compressed_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse_compressed_tensor](https://pytorch.org/docs/2.7/generated/torch.sparse_compressed_tensor.html)

**Support Status**:
<!-- end id65 -->

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->

<!-- npu="910b,A3" id58 -->
</div>
<!-- end id58 -->
