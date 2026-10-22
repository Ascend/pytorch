# torch.sparse

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.11/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.11/sparse.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

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

**Native Documentation**: [Tensor.is_sparse_csr](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_sparse_csr.html)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id3 -->

</div>

> <font size="3">to_sparse_coo()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.to_sparse_coo](https://pytorch.org/docs/2.11/generated/torch.Tensor.to_sparse_coo.html)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id6 -->

</div>

> <font size="3">coalesce()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.coalesce](https://pytorch.org/docs/2.11/generated/torch.Tensor.coalesce.html)

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

**Native Documentation**: [Tensor.sparse_resize_](https://pytorch.org/docs/2.11/generated/torch.Tensor.sparse_resize_.html)

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

**Native Documentation**: [Tensor.sparse_resize_and_clear_](https://pytorch.org/docs/2.11/generated/torch.Tensor.sparse_resize_and_clear_.html)

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

**Native Documentation**: [Tensor.is_coalesced](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_coalesced.html)

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

> <font size="3">crow_indices()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.crow_indices](https://pytorch.org/docs/2.11/generated/torch.Tensor.crow_indices.html)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id21 -->

</div>

> <font size="3">col_indices()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.col_indices](https://pytorch.org/docs/2.11/generated/torch.Tensor.col_indices.html)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id24 -->

</div>

> <font size="3">row_indices()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.row_indices](https://pytorch.org/docs/2.11/generated/torch.Tensor.row_indices.html)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id27 -->

</div>

> <font size="3">ccol_indices()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ccol_indices](https://pytorch.org/docs/2.11/generated/torch.Tensor.ccol_indices.html)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id30 -->

</div>

</div>

### <code><i>class</i></code> torch.sparse.check_sparse_tensor_invariants

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse.check_sparse_tensor_invariants](https://pytorch.org/docs/2.11/generated/torch.sparse.check_sparse_tensor_invariants.html)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id33 -->

> <font size="3">disable()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse.check_sparse_tensor_invariants.disable](https://pytorch.org/docs/2.11/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants.disable)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id36 -->

</div>

> <font size="3">enable()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse.check_sparse_tensor_invariants.enable](https://pytorch.org/docs/2.11/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants.enable)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id39 -->

</div>

> <font size="3">is_enabled()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse.check_sparse_tensor_invariants.is_enabled](https://pytorch.org/docs/2.11/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants.is_enabled)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id42 -->

</div>

</div>

### torch.sparse_compressed_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse_compressed_tensor](https://pytorch.org/docs/2.11/generated/torch.sparse_compressed_tensor.html)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id45 -->

</div>

### torch.sparse.as_sparse_gradcheck

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse.as_sparse_gradcheck](https://pytorch.org/docs/2.11/generated/torch.sparse.as_sparse_gradcheck.html)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id48 -->

</div>
