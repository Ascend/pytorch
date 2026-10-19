# torch.nn.functional

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/nn.functional.html).

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

</div>

<div style="display:none;">

## &#8203;torch.nn.functional

</div>

### torch.nn.parallel.data_parallel

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

## Convolution functions

### torch.nn.functional.conv1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.conv2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.conv3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, complex64

</div>

### torch.nn.functional.conv_transpose1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.nn.functional.conv_transpose2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.conv_transpose3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.unfold

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.fold

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

## Pooling functions

### torch.nn.functional.avg_pool1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.avg_pool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.avg_pool3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.functional.max_pool1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.functional.max_pool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- By setting torch_npu.npu.use_compatible_impl(True), the interface is aligned with the community interface of the same name in memory consistency.

</div>

### torch.nn.functional.max_pool3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- Currently, on the NPU, the value of `dilation` only supports being set to 1 or (1,1,1).
- When `return_indices` is True, the data type of the returned `argmax` is int32.

</div>

### torch.nn.functional.max_unpool1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int32, int64

</div>

### torch.nn.functional.max_unpool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32, fp64, uint8, int8, int32, int64
- When `jit_compile=False`, that is, in binary mode, the product of `output_size` must be greater than or equal to the product of H and W of `input`.

</div>

### torch.nn.functional.max_unpool3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.functional.lp_pool1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.lp_pool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.adaptive_max_pool1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.functional.adaptive_max_pool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.functional.adaptive_max_pool3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp32, fp64

</div>

### torch.nn.functional.adaptive_avg_pool1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.adaptive_avg_pool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.adaptive_avg_pool3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.fractional_max_pool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: May fall back to CPU execution.

</div>

### torch.nn.functional.fractional_max_pool3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

## Attention Mechanisms

### torch.nn.functional.scaled_dot_product_attention

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- In the current version, the sdpa (scaled_dot_product_attention) interface is only available as a trial feature. This feature may be adjusted or improved in subsequent versions. Users are advised to pay attention to the iterations of subsequent versions during use.
- `query`, `key`, and `value` only support bf16, fp16, and fp32. All parameter inputs must comply with the following constraints:
- **Constraints:**
- All parameter inputs must comply with the specifications:
  - N of input `query`, `key`, and `value`: Batch size. Currently, only the layout [N, head_num, S(L), E(Ev)] is supported, with a value range of 1 to 2K.
  - The head num of input `query` and the head num of `key`/`value` must be proportional, that is, Nq/Nkv must be a non-zero integer, with a value range of 1 to 256.
  - L of input `query`: Target sequence length, with a value range of 1 to 512K.
  - S of input `key` and `value`: Source sequence length, with a value range of 1 to 512K.
- E of input `query`, `key`, and `value`: Embedding dimension of the query and key, with a value range of 1 to 512.
  - Ev of input `value`: Embedding dimension of the value, which must be equal to E.
  - Input `attn_mask`: Currently supports [N, 1, L, S], [N, head_num, L, S], [1, 1, L, S], [L, S], and bool-type masks that can be broadcast to [N, head_num, L, S], for example, layouts such as [L, 1], [1, S], and [1, 1].
  - When `is_causal` computation is enabled, `attn_mask` must be None. When `is_causal` is not enabled, if valid data is input for `attn_mask`, the input data type must be bool.
  - The data types of input `query`, `key`, and `value` are bf16, fp16, and fp32.
  - By setting torch_npu.npu.use_compatible_impl(True), the MATH backend can be specified according to the SDPA backend selection context.
- Differences from the original interface apart from specification restrictions:
  - The random algorithm on the NPU is implemented using DSA hardware. The algorithm is solidified in the DSA engine, which differs from the GPU algorithm implementation, causing the dropout function to be inconsistent with GPU results.
  - The current interface supports different head num values for input `query` and `key`/`value`, while the native PyTorch interface does not support this.

</div>

## Non-linear activation functions

### torch.nn.functional.threshold

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64
- When the input is of int32 type exceeding 16,777,216 (that is, 2<sup>24</sup>), there will be a loss of precision.

</div>

### torch.nn.functional.threshold_

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64
- When the input is of int32 type exceeding 16,777,216 (that is, 2<sup>24</sup>), there will be a loss of precision.

</div>

### torch.nn.functional.relu

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64

</div>

### torch.nn.functional.relu_

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64

</div>

### torch.nn.functional.hardtanh

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.nn.functional.hardtanh_

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32, int8, int16, int32, int64

</div>

### torch.nn.functional.hardswish

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32
- May fall back to CPU execution

</div>

### torch.nn.functional.relu6

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.nn.functional.elu

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### torch.nn.functional.elu_

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.selu

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32
- In fp16 backward scenarios, compared with GPU, there are accumulated precision errors, which can be avoided in the following way:
- Replace the forward call torch.nn.functional.selu with torch.ops.aten.elu. For example, replace torch.nn.functional.selu(input_x) with torch.ops.aten.elu(input_x, 1.6732632423543772848170429916717, 1.0507009873554804934193349852946).

</div>

### torch.nn.functional.celu

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.leaky_relu

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### torch.nn.functional.leaky_relu_

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32, fp64

</div>

### torch.nn.functional.prelu

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32
- `input` only supports 1-8 dimensions

</div>

### torch.nn.functional.rrelu

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.glu

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### torch.nn.functional.gelu

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- The `approximate` parameter only supports being set to `tanh`.

</div>

### torch.nn.functional.logsigmoid

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.hardshrink

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.softsign

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.nn.functional.softplus

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.softmax

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### torch.nn.functional.softshrink

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.gumbel_softmax

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.functional.log_softmax

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.tanh

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.nn.functional.sigmoid

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nn.functional.hardsigmoid

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.silu

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.mish

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.batch_norm

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32
- `weight` and `bias` only support one-dimensional scenarios.
- The shape of `bias` is 1 dimension, and its length is equal to the length of the channel axis in the `input` parameter.

</div>

### torch.nn.functional.group_norm

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- This API only supports input `input` with 2 or more dimensions.
- The `eps` parameter must be greater than 0.

</div>

### torch.nn.functional.layer_norm

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.normalize

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

## Linear functions

### torch.nn.functional.linear

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.bilinear

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input1`, `input2`, `weight`, `bias` only supports bf16, fp16, fp32

</div>

## Dropout functions

### torch.nn.functional.dropout

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nn.functional.alpha_dropout

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nn.functional.feature_alpha_dropout

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nn.functional.dropout2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- <term>Ascend 950DT</term>: Not Supported (complex64, complex128)

</div>

## Sparse functions

### torch.nn.functional.embedding

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, int32, int64
- The `max_norm` attribute only supports non-negative values.

</div>

### torch.nn.functional.embedding_bag

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.functional.one_hot

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports int64

</div>

## Distance functions

### torch.nn.functional.cosine_similarity

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `x1`, `x2` only supports fp16, fp32

</div>

### torch.nn.functional.pdist

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

## Loss functions

### torch.nn.functional.binary_cross_entropy

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.binary_cross_entropy_with_logits

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.poisson_nll_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, int64
- May fall back to CPU execution

</div>

### torch.nn.functional.cross_entropy

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.ctc_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `log_probs` only supports fp32, fp64
- The length of the target sequence does not support 0, that is, the value of the `target_lengths` attribute cannot contain 0.

</div>

### torch.nn.functional.gaussian_nll_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.nn.functional.kl_div

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- Currently, the `log_target` parameter only supports False.
- Currently, `target` does not support differentiation.

</div>

### torch.nn.functional.l1_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int64

</div>

### torch.nn.functional.mse_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128

</div>

### torch.nn.functional.margin_ranking_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input1`, `input2`, `target` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.multilabel_margin_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32
- The number of elements in input `tensor` cannot exceed 100,000.

</div>

### torch.nn.functional.multilabel_soft_margin_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.functional.nll_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp32
- Each element value in `target` should be greater than or equal to 0 and less than the number of classes of `input`.

</div>

### torch.nn.functional.smooth_l1_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.functional.soft_margin_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, and fp32, and does not support the double, complex64, and complex128 data types.

</div>

### torch.nn.functional.triplet_margin_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.functional.triplet_margin_with_distance_loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

## Vision functions

### torch.nn.functional.pixel_shuffle

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nn.functional.pixel_unshuffle

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.nn.functional.pad

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- When the `mode` attribute is constant, `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool.
- When the `mode` attribute is not constant, `input` only supports fp16, fp32, and fp64.
- Performance may degrade when input `x` has six or more dimensions.

</div>

### torch.nn.functional.interpolate

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64
- `input` only supports nearest、linear、bilinear、bicubic、trilinear、area
- Does not support `scale_factor`.

</div>

### torch.nn.functional.upsample

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32, fp64
- Only supports `mode` = nearest.

</div>

### torch.nn.functional.upsample_nearest

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32, fp64
- Only supports 3-5 dimensions.

</div>

### torch.nn.functional.upsample_bilinear

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.nn.functional.grid_sample

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32, fp64

</div>

### torch.nn.functional.affine_grid

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `theta` only supports fp16, fp32

</div>
