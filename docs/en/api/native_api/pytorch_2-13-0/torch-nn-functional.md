# torch.nn.functional

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.13/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.13/nn.functional.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Convolution functions](#convolution-functions)
- [Pooling functions](#pooling-functions)
- [Attention Mechanisms](#attention-mechanisms)
- [Non-linear activation functions](#non-linear-activation-functions)
- [Linear functions](#linear-functions)
- [Dropout functions](#dropout-functions)
- [Sparse functions](#sparse-functions)
- [Distance functions](#distance-functions)
- [Loss functions](#loss-functions)
- [Vision functions](#vision-functions)
- [Low-Precision functions](#low-precision-functions)

</div>

<div style="display:none;">

## &#8203;torch.nn.functional

</div>

### torch.nn.parallel.data_parallel

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.parallel.data_parallel](https://pytorch.org/docs/2.13/generated/torch.nn.functional.torch.nn.parallel.data_parallel.html#torch.nn.parallel.data_parallel)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

</div>

## Convolution functions

### torch.nn.functional.conv1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.conv1d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.conv1d.html)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.conv2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.conv2d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.conv2d.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.conv3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.conv3d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.conv3d.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, complex64

</div>

### torch.nn.functional.conv_transpose1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.conv_transpose1d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.conv_transpose1d.html)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.nn.functional.conv_transpose2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.conv_transpose2d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.conv_transpose2d.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.conv_transpose3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.conv_transpose3d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.conv_transpose3d.html)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id21 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.unfold

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.unfold](https://pytorch.org/docs/2.13/generated/torch.nn.functional.unfold.html)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.fold

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.fold](https://pytorch.org/docs/2.13/generated/torch.nn.functional.fold.html)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id27 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

## Pooling functions

### torch.nn.functional.avg_pool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.avg_pool1d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.avg_pool1d.html)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.avg_pool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.avg_pool2d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.avg_pool2d.html)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id33 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.avg_pool3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.avg_pool3d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.avg_pool3d.html)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id36 -->

</div>

### torch.nn.functional.max_pool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.max_pool1d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.max_pool1d.html)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32
- `dilation` only supports 1
- By setting `torch_npu.npu.use_compatible_impl(True)`, the interface is aligned with the PyTorch interface of the same name in memory consistency, for example:

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

- When `return_indices` is True, the data type of the returned `argmax` is int32.

</div>

### torch.nn.functional.max_pool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.max_pool2d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.max_pool2d.html)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- The value of `dilation` currently only supports 1 or (1,1) on the NPU
- By setting `torch_npu.npu.use_compatible_impl(True)`, the interface is aligned with the PyTorch interface of the same name in memory consistency, for example:

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

- When `return_indices` is True, the data type of the returned `argmax` is int32.

</div>

### torch.nn.functional.max_pool3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.max_pool3d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.max_pool3d.html)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id45 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- Currently, on the NPU, the value of `dilation` only supports being set to 1 or (1,1,1).
- When `return_indices` is True, the data type of the returned `argmax` is int32.

</div>

### torch.nn.functional.max_unpool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.max_unpool1d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.max_unpool1d.html)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id48 -->

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int32, int64

</div>

### torch.nn.functional.max_unpool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.max_unpool2d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.max_unpool2d.html)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id51 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32, fp64, uint8, int8, int32, int64
- When `jit_compile=False`, that is, in binary mode, the product of `output_size` must be greater than or equal to the product of H and W of `input`.

</div>

### torch.nn.functional.max_unpool3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.max_unpool3d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.max_unpool3d.html)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id54 -->

</div>

### torch.nn.functional.lp_pool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.lp_pool1d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.lp_pool1d.html)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.lp_pool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.lp_pool2d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.lp_pool2d.html)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id60 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.adaptive_max_pool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.adaptive_max_pool1d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.adaptive_max_pool1d.html)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id63 -->

</div>

### torch.nn.functional.adaptive_max_pool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.adaptive_max_pool2d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.adaptive_max_pool2d.html)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id66 -->

</div>

### torch.nn.functional.adaptive_max_pool3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.adaptive_max_pool3d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.adaptive_max_pool3d.html)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id69 -->

**Constraints & Notes**: `input` only supports fp32, fp64

</div>

### torch.nn.functional.adaptive_avg_pool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.adaptive_avg_pool1d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.adaptive_avg_pool1d.html)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id72 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.adaptive_avg_pool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.adaptive_avg_pool2d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.adaptive_avg_pool2d.html)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.adaptive_avg_pool3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.adaptive_avg_pool3d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.adaptive_avg_pool3d.html)

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id78 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.fractional_max_pool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.fractional_max_pool2d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.fractional_max_pool2d.html)

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id81 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

### torch.nn.functional.fractional_max_pool3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.fractional_max_pool3d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.fractional_max_pool3d.html)

**Support Status**:

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id84 -->

</div>

## Attention Mechanisms

### torch.nn.functional.scaled_dot_product_attention

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention)

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id87 -->

**Constraints & Notes**:

- `query`, `key`, and `value` only support bf16, fp16, and fp32. All parameter inputs must comply with the following constraints:
- **Constraints:**
- All parameter inputs must comply with the specifications:
  - N of input `query`, `key`, and `value`: batch size. Currently, only the layout [N, head_num, S(L), E(Ev)] is supported, with a value range of 1 to 2K.
  - The head num of input `query` and the head num of `key`/`value` must be proportional, that is, Nq/Nkv must be a non-zero integer, with a value range of 1 to 256.
  - L of input `query`: Target sequence length, with a value range of 1 to 512K.
  - S of input `key` and `value`: Source sequence length, with a value range of 1 to 512K.
  - E of input `query`, `key`, and `value`: Embedding dimension of the query and key, with a value range of 1 to 512.
  - Ev of input `value`: Embedding dimension of the value, which must be equal to E.
  - Input `attn_mask`: currently supports [N, 1, L, S], [N, head_num, L, S], [1, 1, L, S], [L, S], and bool-type masks that can be broadcast to [N, head_num, L, S], for example, layouts such as [L, 1], [1, S], and [1, 1].
  - When `is_causal` computation is enabled, `attn_mask` must be None. When `is_causal` is not enabled, if valid data is input for `attn_mask`, the input data type must be bool.
  - The data types of input `query`, `key`, and `value` are bf16, fp16, and fp32.
  - By setting `torch_npu.npu.use_compatible_impl(True)`, the MATH backend can be specified according to the SDPA backend selection context, for example:

    ```python
    import torch_npu
    torch_npu.npu.use_compatible_impl(True)
    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    ```

- Differences from the original interface apart from specification restrictions:
  - The random algorithm on the NPU is implemented using DSA hardware. The algorithm is solidified in the DSA engine, which differs from the GPU algorithm implementation, causing the dropout function to be inconsistent with GPU results.
  - The current interface supports different head num values for input `query` and `key`/`value`, while the native PyTorch interface does not support this.

</div>

## Non-linear activation functions

### torch.nn.functional.threshold

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.threshold](https://pytorch.org/docs/2.13/generated/torch.nn.functional.threshold.html)

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id90 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64
- When `input` is of the int32 type exceeding 16,777,216 (that is, 2<sup>24</sup>), precision will be lost

</div>

### torch.nn.functional.threshold_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.threshold_](https://pytorch.org/docs/2.13/generated/torch.nn.functional.threshold_.html)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id93 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64
- When `input` is of the int32 type exceeding 16,777,216 (that is, 2<sup>24</sup>), precision will be lost

</div>

### torch.nn.functional.relu

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.relu](https://pytorch.org/docs/2.13/generated/torch.nn.functional.relu.html)

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id96 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64

</div>

### torch.nn.functional.relu_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.relu_](https://pytorch.org/docs/2.13/generated/torch.nn.functional.relu_.html)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id99 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64

</div>

### torch.nn.functional.hardtanh

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.hardtanh](https://pytorch.org/docs/2.13/generated/torch.nn.functional.hardtanh.html)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id102 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.nn.functional.hardtanh_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.hardtanh_](https://pytorch.org/docs/2.13/generated/torch.nn.functional.hardtanh_.html)

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id105 -->

**Constraints & Notes**: `input` only supports fp16, fp32, int8, int16, int32, int64

</div>

### torch.nn.functional.hardswish

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.hardswish](https://pytorch.org/docs/2.13/generated/torch.nn.functional.hardswish.html)

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id108 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32
- May fall back to CPU execution

</div>

### torch.nn.functional.relu6

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.relu6](https://pytorch.org/docs/2.13/generated/torch.nn.functional.relu6.html)

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id111 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.nn.functional.elu

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.elu](https://pytorch.org/docs/2.13/generated/torch.nn.functional.elu.html)

**Support Status**:

<!-- npu="910b" id112 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id114 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### torch.nn.functional.elu_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.elu_](https://pytorch.org/docs/2.13/generated/torch.nn.functional.elu_.html)

**Support Status**:

<!-- npu="910b" id115 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id117 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.selu

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.selu](https://pytorch.org/docs/2.13/generated/torch.nn.functional.selu.html)

**Support Status**:

<!-- npu="910b" id118 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id120 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32
- In fp16 backward scenarios, compared with GPU, there are accumulated precision errors, which can be avoided in the following way:
- Replace the forward call torch.nn.functional.selu with torch.ops.aten.elu. For example, replace torch.nn.functional.selu(input_x) with torch.ops.aten.elu(input_x, 1.6732632423543772848170429916717, 1.0507009873554804934193349852946).

</div>

### torch.nn.functional.celu

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.celu](https://pytorch.org/docs/2.13/generated/torch.nn.functional.celu.html)

**Support Status**:

<!-- npu="910b" id121 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id123 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.leaky_relu

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.leaky_relu](https://pytorch.org/docs/2.13/generated/torch.nn.functional.leaky_relu.html)

**Support Status**:

<!-- npu="910b" id124 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id126 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### torch.nn.functional.leaky_relu_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.leaky_relu_](https://pytorch.org/docs/2.13/generated/torch.nn.functional.leaky_relu_.html)

**Support Status**:

<!-- npu="910b" id127 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id129 -->

**Constraints & Notes**: `input` only supports fp16, fp32, fp64

</div>

### torch.nn.functional.prelu

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.prelu](https://pytorch.org/docs/2.13/generated/torch.nn.functional.prelu.html)

**Support Status**:

<!-- npu="910b" id130 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id132 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32
- `input` only supports 1-8 dimensions

</div>

### torch.nn.functional.rrelu

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.rrelu](https://pytorch.org/docs/2.13/generated/torch.nn.functional.rrelu.html)

**Support Status**:

<!-- npu="910b" id133 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id135 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.glu

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.glu](https://pytorch.org/docs/2.13/generated/torch.nn.functional.glu.html)

**Support Status**:

<!-- npu="910b" id136 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id138 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### torch.nn.functional.gelu

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.gelu](https://pytorch.org/docs/2.13/generated/torch.nn.functional.gelu.html)

**Support Status**:

<!-- npu="910b" id139 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id141 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- The `approximate` parameter only supports being set to `tanh`.

</div>

### torch.nn.functional.logsigmoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.logsigmoid](https://pytorch.org/docs/2.13/generated/torch.nn.functional.logsigmoid.html)

**Support Status**:

<!-- npu="910b" id142 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id144 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.hardshrink

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.hardshrink](https://pytorch.org/docs/2.13/generated/torch.nn.functional.hardshrink.html)

**Support Status**:

<!-- npu="910b" id145 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id147 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.softsign

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.softsign](https://pytorch.org/docs/2.13/generated/torch.nn.functional.softsign.html)

**Support Status**:

<!-- npu="910b" id148 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id150 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.nn.functional.softplus

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.softplus](https://pytorch.org/docs/2.13/generated/torch.nn.functional.softplus.html)

**Support Status**:

<!-- npu="910b" id151 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id153 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.softmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.softmax](https://pytorch.org/docs/2.13/generated/torch.nn.functional.softmax.html)

**Support Status**:

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id156 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### torch.nn.functional.softshrink

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.softshrink](https://pytorch.org/docs/2.13/generated/torch.nn.functional.softshrink.html)

**Support Status**:

<!-- npu="910b" id157 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id159 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.gumbel_softmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.gumbel_softmax](https://pytorch.org/docs/2.13/generated/torch.nn.functional.gumbel_softmax.html)

**Support Status**:

<!-- npu="910b" id160 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id162 -->

</div>

### torch.nn.functional.log_softmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.log_softmax](https://pytorch.org/docs/2.13/generated/torch.nn.functional.log_softmax.html)

**Support Status**:

<!-- npu="910b" id163 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id165 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.tanh

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.tanh](https://pytorch.org/docs/2.13/generated/torch.nn.functional.tanh.html)

**Support Status**:

<!-- npu="910b" id166 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id168 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.nn.functional.sigmoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.sigmoid](https://pytorch.org/docs/2.13/generated/torch.nn.functional.sigmoid.html)

**Support Status**:

<!-- npu="910b" id169 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id171 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nn.functional.hardsigmoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.hardsigmoid](https://pytorch.org/docs/2.13/generated/torch.nn.functional.hardsigmoid.html)

**Support Status**:

<!-- npu="910b" id172 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id174 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.silu

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.silu](https://pytorch.org/docs/2.13/generated/torch.nn.functional.silu.html)

**Support Status**:

<!-- npu="910b" id175 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id177 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.mish

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.mish](https://pytorch.org/docs/2.13/generated/torch.nn.functional.mish.html)

**Support Status**:

<!-- npu="910b" id178 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id180 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.batch_norm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.batch_norm](https://pytorch.org/docs/2.13/generated/torch.nn.functional.batch_norm.html)

**Support Status**:

<!-- npu="910b" id181 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id183 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32
- `weight` and `bias` only support one-dimensional scenarios.
- The shape of `bias` is 1 dimension, and its length is equal to the length of the channel axis in the `input` parameter.

</div>

### torch.nn.functional.group_norm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.group_norm](https://pytorch.org/docs/2.13/generated/torch.nn.functional.group_norm.html)

**Support Status**:

<!-- npu="910b" id184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id186 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- This API only supports input `input` with 2 or more dimensions.
- The `eps` parameter must be greater than 0.

</div>

### torch.nn.functional.layer_norm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.layer_norm](https://pytorch.org/docs/2.13/generated/torch.nn.functional.layer_norm.html)

**Support Status**:

<!-- npu="910b" id187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id189 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.normalize

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.normalize](https://pytorch.org/docs/2.13/generated/torch.nn.functional.normalize.html)

**Support Status**:

<!-- npu="910b" id190 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id192 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

## Linear functions

### torch.nn.functional.linear

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.linear](https://pytorch.org/docs/2.13/generated/torch.nn.functional.linear.html)

**Support Status**:

<!-- npu="910b" id193 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id195 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.bilinear

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.bilinear](https://pytorch.org/docs/2.13/generated/torch.nn.functional.bilinear.html)

**Support Status**:

<!-- npu="910b" id196 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id198 -->

**Constraints & Notes**: `input1`, `input2`, `weight`, and `bias` only support bf16, fp16, and fp32

</div>

## Dropout functions

### torch.nn.functional.dropout

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.dropout](https://pytorch.org/docs/2.13/generated/torch.nn.functional.dropout.html)

**Support Status**:

<!-- npu="910b" id199 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id201 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nn.functional.alpha_dropout

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.alpha_dropout](https://pytorch.org/docs/2.13/generated/torch.nn.functional.alpha_dropout.html)

**Support Status**:

<!-- npu="910b" id202 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id204 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nn.functional.feature_alpha_dropout

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.feature_alpha_dropout](https://pytorch.org/docs/2.13/generated/torch.nn.functional.feature_alpha_dropout.html)

**Support Status**:

<!-- npu="910b" id205 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id207 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nn.functional.dropout2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.dropout2d](https://pytorch.org/docs/2.13/generated/torch.nn.functional.dropout2d.html)

**Support Status**:

<!-- npu="910b" id208 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id210 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

<!-- npu="950" id211 -->
- <term>Ascend 950DT products</term>: Not Supported for complex64, complex128
<!-- end id211 -->

</div>

## Sparse functions

### torch.nn.functional.embedding

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.embedding](https://pytorch.org/docs/2.13/generated/torch.nn.functional.embedding.html)

**Support Status**:

<!-- npu="910b" id212 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id212 -->
<!-- npu="A3" id213 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id213 -->
<!-- npu="950" id214 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id214 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, int32, int64
- The `max_norm` attribute only supports non-negative values.

</div>

### torch.nn.functional.embedding_bag

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.embedding_bag](https://pytorch.org/docs/2.13/generated/torch.nn.functional.embedding_bag.html)

**Support Status**:

<!-- npu="910b" id215 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id215 -->
<!-- npu="A3" id216 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id216 -->
<!-- npu="950" id217 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id217 -->

</div>

### torch.nn.functional.one_hot

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.one_hot](https://pytorch.org/docs/2.13/generated/torch.nn.functional.one_hot.html)

**Support Status**:

<!-- npu="910b" id218 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id218 -->
<!-- npu="A3" id219 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id219 -->
<!-- npu="950" id220 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id220 -->

**Constraints & Notes**: `input` only supports int32, int64

</div>

## Distance functions

### torch.nn.functional.cosine_similarity

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.cosine_similarity](https://pytorch.org/docs/2.13/generated/torch.nn.functional.cosine_similarity.html)

**Support Status**:

<!-- npu="910b" id221 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id221 -->
<!-- npu="A3" id222 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id222 -->
<!-- npu="950" id223 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id223 -->

**Constraints & Notes**: `x1` and `x2` only support fp16 and fp32

</div>

### torch.nn.functional.pdist

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.pdist](https://pytorch.org/docs/2.13/generated/torch.nn.functional.pdist.html)

**Support Status**:

<!-- npu="910b" id224 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id224 -->
<!-- npu="A3" id225 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id225 -->
<!-- npu="950" id226 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id226 -->

</div>

## Loss functions

### torch.nn.functional.binary_cross_entropy

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.binary_cross_entropy](https://pytorch.org/docs/2.13/generated/torch.nn.functional.binary_cross_entropy.html)

**Support Status**:

<!-- npu="910b" id227 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id227 -->
<!-- npu="A3" id228 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id228 -->
<!-- npu="950" id229 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id229 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.binary_cross_entropy_with_logits

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.binary_cross_entropy_with_logits](https://pytorch.org/docs/2.13/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)

**Support Status**:

<!-- npu="910b" id230 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id230 -->
<!-- npu="A3" id231 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id231 -->
<!-- npu="950" id232 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id232 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.poisson_nll_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.poisson_nll_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.poisson_nll_loss.html)

**Support Status**:

<!-- npu="910b" id233 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id233 -->
<!-- npu="A3" id234 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id234 -->
<!-- npu="950" id235 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id235 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, int64
- May fall back to CPU execution

</div>

### torch.nn.functional.cross_entropy

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.cross_entropy](https://pytorch.org/docs/2.13/generated/torch.nn.functional.cross_entropy.html)

**Support Status**:

<!-- npu="910b" id236 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id236 -->
<!-- npu="A3" id237 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id237 -->
<!-- npu="950" id238 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id238 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.ctc_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.ctc_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.ctc_loss.html)

**Support Status**:

<!-- npu="910b" id239 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id239 -->
<!-- npu="A3" id240 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id240 -->
<!-- npu="950" id241 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id241 -->

**Constraints & Notes**:

- `log_probs` only supports fp32, fp64
- The length of the target sequence does not support 0, that is, the value of the `target_lengths` attribute cannot contain 0.

</div>

### torch.nn.functional.gaussian_nll_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.gaussian_nll_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.gaussian_nll_loss.html)

**Support Status**:

<!-- npu="910b" id242 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id242 -->
<!-- npu="A3" id243 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id243 -->
<!-- npu="950" id244 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id244 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.nn.functional.kl_div

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.kl_div](https://pytorch.org/docs/2.13/generated/torch.nn.functional.kl_div.html)

**Support Status**:

<!-- npu="910b" id245 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id245 -->
<!-- npu="A3" id246 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id246 -->
<!-- npu="950" id247 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id247 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- Currently, the `log_target` parameter only supports False.
- Currently, `target` does not support differentiation.

</div>

### torch.nn.functional.l1_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.l1_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.l1_loss.html)

**Support Status**:

<!-- npu="910b" id248 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id248 -->
<!-- npu="A3" id249 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id249 -->
<!-- npu="950" id250 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id250 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int64

</div>

### torch.nn.functional.mse_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.mse_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.mse_loss.html)

**Support Status**:

<!-- npu="910b" id251 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id251 -->
<!-- npu="A3" id252 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id252 -->
<!-- npu="950" id253 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id253 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.margin_ranking_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.margin_ranking_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.margin_ranking_loss.html)

**Support Status**:

<!-- npu="910b" id254 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id254 -->
<!-- npu="A3" id255 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id255 -->
<!-- npu="950" id256 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id256 -->

**Constraints & Notes**: `input1`, `input2`, and `target` only support bf16, fp16, and fp32

</div>

### torch.nn.functional.multilabel_margin_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.multilabel_margin_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.multilabel_margin_loss.html)

**Support Status**:

<!-- npu="910b" id257 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id257 -->
<!-- npu="A3" id258 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id258 -->
<!-- npu="950" id259 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id259 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32
- The number of elements in input `tensor` cannot exceed 100,000.

</div>

### torch.nn.functional.multilabel_soft_margin_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.multilabel_soft_margin_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.multilabel_soft_margin_loss.html)

**Support Status**:

<!-- npu="910b" id260 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id260 -->
<!-- npu="A3" id261 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id261 -->
<!-- npu="950" id262 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id262 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.nll_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.nll_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.nll_loss.html)

**Support Status**:

<!-- npu="910b" id263 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id263 -->
<!-- npu="A3" id264 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id264 -->
<!-- npu="950" id265 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id265 -->

**Constraints & Notes**:

- `input` only supports fp32
- The value of each element in `target` must be ≥0 and less than the number of classes of `input`. Otherwise, an incorrect result is returned instead of an error

</div>

### torch.nn.functional.smooth_l1_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.smooth_l1_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.smooth_l1_loss.html)

**Support Status**:

<!-- npu="910b" id266 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id266 -->
<!-- npu="A3" id267 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id267 -->
<!-- npu="950" id268 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id268 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.soft_margin_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.soft_margin_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.soft_margin_loss.html)

**Support Status**:

<!-- npu="910b" id269 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id269 -->
<!-- npu="A3" id270 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id270 -->
<!-- npu="950" id271 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id271 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, and does not support double, complex64, complex128 data types

</div>

### torch.nn.functional.triplet_margin_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.triplet_margin_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.triplet_margin_loss.html)

**Support Status**:

<!-- npu="910b" id272 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id272 -->
<!-- npu="A3" id273 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id273 -->
<!-- npu="950" id274 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id274 -->

</div>

### torch.nn.functional.triplet_margin_with_distance_loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.triplet_margin_with_distance_loss](https://pytorch.org/docs/2.13/generated/torch.nn.functional.triplet_margin_with_distance_loss.html)

**Support Status**:

<!-- npu="910b" id275 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id275 -->
<!-- npu="A3" id276 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id276 -->
<!-- npu="950" id277 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id277 -->

</div>

## Vision functions

### torch.nn.functional.pixel_shuffle

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.pixel_shuffle](https://pytorch.org/docs/2.13/generated/torch.nn.functional.pixel_shuffle.html)

**Support Status**:

<!-- npu="910b" id278 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id278 -->
<!-- npu="A3" id279 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id279 -->
<!-- npu="950" id280 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id280 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nn.functional.pixel_unshuffle

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.pixel_unshuffle](https://pytorch.org/docs/2.13/generated/torch.nn.functional.pixel_unshuffle.html)

**Support Status**:

<!-- npu="910b" id281 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id281 -->
<!-- npu="A3" id282 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id282 -->
<!-- npu="950" id283 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id283 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.nn.functional.pad

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.pad](https://pytorch.org/docs/2.13/generated/torch.nn.functional.pad.html)

**Support Status**:

<!-- npu="910b" id284 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id284 -->
<!-- npu="A3" id285 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id285 -->
<!-- npu="950" id286 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id286 -->

**Constraints & Notes**:

- When the `mode` attribute is constant, `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool.
- When the `mode` attribute is not constant, `input` only supports fp16, fp32, and fp64.
- Performance may degrade when input `x` has six or more dimensions.

</div>

### torch.nn.functional.interpolate

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.interpolate](https://pytorch.org/docs/2.13/generated/torch.nn.functional.interpolate.html)

**Support Status**:

<!-- npu="910b" id287 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id287 -->
<!-- npu="A3" id288 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id288 -->
<!-- npu="950" id289 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id289 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64
- Supports nearest, linear, bilinear, bicubic, trilinear, and area
- Does not support `scale_factor`.

</div>

### torch.nn.functional.upsample

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.upsample](https://pytorch.org/docs/2.13/generated/torch.nn.functional.upsample.html)

**Support Status**:

<!-- npu="910b" id290 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id290 -->
<!-- npu="A3" id291 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id291 -->
<!-- npu="950" id292 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id292 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32, fp64
- Only `mode` = nearest is supported, for example:

  ```python
  out = torch.nn.functional.upsample(x, size=(256, 256), mode='nearest')
  ```

</div>

### torch.nn.functional.upsample_nearest

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.upsample_nearest](https://pytorch.org/docs/2.13/generated/torch.nn.functional.upsample_nearest.html)

**Support Status**:

<!-- npu="910b" id293 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id293 -->
<!-- npu="A3" id294 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id294 -->
<!-- npu="950" id295 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id295 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- `input` only supports 3-5 dimensions

</div>

### torch.nn.functional.upsample_bilinear

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.upsample_bilinear](https://pytorch.org/docs/2.13/generated/torch.nn.functional.upsample_bilinear.html)

**Support Status**:

<!-- npu="910b" id296 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id296 -->
<!-- npu="A3" id297 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id297 -->
<!-- npu="950" id298 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id298 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.grid_sample

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.grid_sample](https://pytorch.org/docs/2.13/generated/torch.nn.functional.grid_sample.html)

**Support Status**:

<!-- npu="910b" id299 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id299 -->
<!-- npu="A3" id300 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id300 -->
<!-- npu="950" id301 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id301 -->

**Constraints & Notes**: `input` only supports fp16, fp32, fp64

</div>

### torch.nn.functional.affine_grid

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.affine_grid](https://pytorch.org/docs/2.13/generated/torch.nn.functional.affine_grid.html)

**Support Status**:

<!-- npu="910b" id302 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id302 -->
<!-- npu="A3" id303 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id303 -->
<!-- npu="950" id304 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id304 -->

**Constraints & Notes**: `theta` only supports fp16, fp32

</div>

## Low-Precision functions

### torch.nn.functional.scaled_mm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.scaled_mm](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_mm.html)

**Support Status**:

<!-- npu="910b" id305 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id305 -->
<!-- npu="A3" id306 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id306 -->
<!-- npu="950" id307 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id307 -->

**Constraints & Notes**: `input` only supports the layout with ScalingType tensorwise, rowwise, and BlockWise1x128 in fp8 mode, and BlockWise1x32 in mxfp8 mode. mxfp8 complies with the [aclnnQuantMatmulV5](https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md) requirements (for `scale_a` and `scale_b`, see the constraint description)

</div>

### torch.nn.functional.scaled_grouped_mm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.functional.scaled_grouped_mm](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_grouped_mm.html)

**Support Status**:

<!-- npu="910b" id308 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id308 -->
<!-- npu="A3" id309 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id309 -->
<!-- npu="950" id310 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id310 -->

**Constraints & Notes**: `input` only supports the layout with ScalingType rowwise in fp8 mode and mxfp8 mode. mxfp8 complies with the [aclnnGroupedMatmulV5](https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md) requirements (for `scale_a` and `scale_b`, see the constraint description)

</div>
