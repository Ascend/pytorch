# torch.Tensor

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.10/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.10/tensors.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Tensor class reference](#tensor-class-reference)

</div>

<div style="display:none;">

## &#8203;torch.Tensor

</div>

## Tensor class reference

### <code><i>class</i></code> torch.Tensor

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">T</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">H</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">mT</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">mH</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">new_tensor()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">new_full()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports int64

</div>

> <font size="3">new_empty()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">new_ones()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">new_zeros()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">is_quantized()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">is_meta()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">device()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">grad()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">ndim()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">real()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">imag()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">nbytes()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">itemsize()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">abs()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">abs_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">absolute()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">absolute_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">acos()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool
- May fall back to CPU execution

</div>

> <font size="3">acos_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">arccos()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">arccos_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">add_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">addbmm()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">addbmm_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">addcdiv()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, int64
- The int64 type does not support simultaneous broadcasting of three `tensor`s

</div>

> <font size="3">addcdiv_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>: `self` only supports bf16, fp16, fp32, fp64
- <term>training products</term>: `self` only supports fp16, fp32, fp64
- The int64 type does not support simultaneous broadcasting of three `tensor`s

</div>

> <font size="3">addcmul()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, int64
- The int64 type does not support simultaneous broadcasting of three `tensor`s

</div>

> <font size="3">addcmul_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64
- <term>training products</term>: `self` only supports fp16, fp32, fp64, uint8, int8, int32, int64
- The int64 type does not support simultaneous broadcasting of three `tensor`s

</div>

> <font size="3">addmm()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">addmm_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">sspaddmm()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">addmv()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">addmv_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">addr()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">addr_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">adjoint()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">allclose()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">amax()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">amin()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">aminmax()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">angle()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64

</div>

> <font size="3">apply_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: CPU-only

</div>

> <font size="3">argmax()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">argmin()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">argsort()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64
- For <term>Ascend 950DT</term>, due to underlying implementation limitations, "stable" only supports True. If it is set to False, it will be automatically modified to True at execution time.

</div>

> <font size="3">argwhere()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">asin()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">asin_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">arcsin()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">arcsin_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">as_strided()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">atan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">atan_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">arctan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">arctan_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">atan2()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">atan2_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32
- May fall back to CPU execution

</div>

> <font size="3">arctan2()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">arctan2_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">all()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">any()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">backward()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">baddbmm()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">baddbmm_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">bernoulli()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32
- May fall back to CPU execution

</div>

> <font size="3">bernoulli_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">bincount()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports uint8, int8, int16, int32, int64
- The dimension of `weights` must be the same as that of `input`

</div>

> <font size="3">bitwise_not()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_not_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_and()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_and_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_or()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_or_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_xor()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_xor_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bmm()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">bool()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">byte()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">broadcast_to()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">ceil()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">ceil_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">char()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">chunk()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">clamp()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">clamp_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">clip()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">clip_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">contiguous()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">copy_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- int16 does not support inputs with more than 5 dimensions

</div>

> <font size="3">conj()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">resolve_conj()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">resolve_neg()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">copysign()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool
- May fall back to CPU execution

</div>

> <font size="3">cos()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">cos_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, complex64, complex128

</div>

> <font size="3">cosh()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">cosh_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">count_nonzero()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">cov()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int16, int32, int64

</div>

> <font size="3">acosh()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- May fall back to CPU execution

</div>

> <font size="3">acosh_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">arccosh()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">arccosh_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">cross()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128
- The shapes of the two inputs must be consistent

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: The NPU equivalent interface is Tensor.npu, and its `memory_format` parameter only supports torch.contiguous_format.

</div>

> <font size="3">cummax()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool
- May fall back to CPU execution

</div>

> <font size="3">cummin()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">cumsum()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- `self` only supports Named Tensor

</div>

> <font size="3">cumsum_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int64, bool

</div>

> <font size="3">chalf()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">cfloat()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">cdouble()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">data_ptr()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">deg2rad()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">dequantize()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">dense_dim()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">detach()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">detach_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">diag()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64

</div>

> <font size="3">diag_embed()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">diagflat()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64

</div>

> <font size="3">diagonal()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">diagonal_scatter()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">fill_diagonal_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">diff()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">dim()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">dim_order()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">dist()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: <term>Ascend 950DT</term>: does not support fp64, complex64, complex128

</div>

> <font size="3">div()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">div_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64

</div>

> <font size="3">divide()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">divide_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">dot()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- Some interfaces on the NPU currently do not support the double type. For compatibility, fp32 is returned by default. After support is completed, fp64 will be returned properly.
- May fall back to CPU execution

</div>

> <font size="3">dsplit()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">element_size()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">eq()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">eq_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">equal()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">erf()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int64, bool

</div>

> <font size="3">erf_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">erfc()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int64, bool

</div>

> <font size="3">erfc_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">erfinv()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">erfinv_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32

</div>

> <font size="3">exp()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int64, bool, complex64, complex128

</div>

> <font size="3">exp_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, complex64, complex128

</div>

> <font size="3">expm1()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int64, bool

</div>

> <font size="3">expm1_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">expand_as()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">exponential_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64

</div>

> <font size="3">fix()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">fix_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">fill_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">flatten()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">flip()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">fliplr()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">flipud()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">float_power()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex128

</div>

> <font size="3">float_power_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports double

</div>

> <font size="3">floor()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">floor_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">floor_divide()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">floor_divide_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">fmod()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int32, int64

</div>

> <font size="3">fmod_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int32, int64

</div>

> <font size="3">frac()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">frac_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">gather()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, int64
- The dimension of `index` must be the same as that of `input`
- For <term>Ascend 950DT</term>, due to hardware differences, when indexes are duplicated, precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>.

</div>

> <font size="3">ge()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">ge_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">greater_equal()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">greater_equal_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">geometric_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">ger()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">get_device()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">gt()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">gt_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">greater()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">greater_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">hardshrink()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">heaviside()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">histc()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">hsplit()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">index_add_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int64, bool

</div>

> <font size="3">index_add()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">index_copy_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int16, int32, bool

</div>

> <font size="3">index_copy()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">index_fill_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int32, int64, bool

</div>

> <font size="3">index_fill()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int32, bool

</div>

> <font size="3">index_put_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports int64
- For <term>Ascend 950DT</term>, due to hardware differences, when indexes are duplicated, precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>.

</div>

> <font size="3">index_put()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports int64
- For <term>Ascend 950DT</term>, due to hardware differences, when indexes are duplicated, precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>.

</div>

> <font size="3">index_reduce_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">index_reduce()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">index_select()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int16, int32, int64, bool

</div>

> <font size="3">indices()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">inner()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">int()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">int_repr()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">isclose()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int32, int64, bool

</div>

> <font size="3">isfinite()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">isinf()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">isposinf()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">isneginf()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">isnan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">is_contiguous()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_complex()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">is_conj()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">is_floating_point()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_inference()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_leaf()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_set_to()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_shared()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">is_signed()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_sparse()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">isreal()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">item()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">kthvalue()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int32

</div>

> <font size="3">ldexp()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">ldexp_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">le()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">le_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">less_equal()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">less_equal_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">lerp()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">lerp_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">log()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int64, bool, complex64, complex128

</div>

> <font size="3">log_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, complex64, complex128

</div>

> <font size="3">log10()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">log10_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">log1p()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">log1p_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">log2()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">log2_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">logaddexp()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int16, int32, int64, bool

</div>

> <font size="3">logaddexp2()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">logsumexp()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">logical_and()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">logical_and_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">logical_not()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">logical_not_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">logical_or()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">logical_or_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">logical_xor()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- May fall back to CPU execution

</div>

> <font size="3">logical_xor_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">logit()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32
- When the value of `eps` is greater than 1, the output is nan; when the value of `eps` is 1, the output is inf

</div>

> <font size="3">logit_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32
- When the value of `eps` is greater than 1, the output is nan; when the value of `eps` is 1, the output is inf

</div>

> <font size="3">long()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">lt()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">lt_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">less()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">less_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">as_subclass()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">map_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: CPU-only

</div>

> <font size="3">masked_scatter_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32, int64, bool

</div>

> <font size="3">masked_scatter()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">masked_fill_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int8, int32, int64, bool

</div>

> <font size="3">masked_fill()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int64, bool

</div>

> <font size="3">masked_select()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32, bool

</div>

> <font size="3">matmul()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32
- `self` only supports Named Tensor

</div>

> <font size="3">matrix_power()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">max()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int64, bool

</div>

> <font size="3">maximum()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, complex64, complex128

</div>

> <font size="3">nanmean()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">median()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64
- When `input` is bf16, `dim` cannot be a dimension whose axis value is 1 in `input`

</div>

> <font size="3">min()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int64, bool

</div>

> <font size="3">minimum()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">mm()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">smm()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">movedim()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">moveaxis()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">msort()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">mul()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">mul_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">multiply()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">multiply_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">multinomial()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">nansum()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool
- <term>Ascend 950DT</term>: Not Supported (uint8, int8, int16, int32, int64, bool)

</div>

> <font size="3">narrow()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">narrow_copy()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">ndimension()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">nan_to_num()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: <term>Ascend 950DT</term>: does not support fp64, complex64, complex128

</div>

> <font size="3">nan_to_num_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">ne()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">ne_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">nextafter_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: Falls back to CPU execution

</div>

> <font size="3">not_equal()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- May fall back to CPU execution

</div>

> <font size="3">not_equal_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">neg()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int8, int32, int64, complex64, complex128

</div>

> <font size="3">neg_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, int8, int32, int64, complex64, complex128
- May fall back to CPU execution

</div>

> <font size="3">negative()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int8, int32, int64, complex64, complex128

</div>

> <font size="3">negative_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int8, int32, int64, complex64, complex128

</div>

> <font size="3">nelement()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">nonzero()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool
- The nan scenario is not supported

</div>

> <font size="3">norm()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64
- <term>Ascend 950DT</term>: Not Supported (fp64)

</div>

> <font size="3">normal_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32
- May fall back to CPU execution

</div>

> <font size="3">numel()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">numpy()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">outer()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">permute()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">positive()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128

</div>

> <font size="3">pow()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, int16, int64

</div>

> <font size="3">pow_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, int64

</div>

> <font size="3">prod()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">put_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128

</div>

> <font size="3">qscheme()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">quantile()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">rad2deg()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">random_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">ravel()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">reciprocal()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">reciprocal_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">record_stream()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_hook()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_post_accumulate_grad_hook()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">remainder()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int32, int64

</div>

> <font size="3">remainder_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int32, int64

</div>

> <font size="3">repeat()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">repeat_interleave()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, int16, int32, bool
- After the input tensor is repeated to obtain the output, the number of elements in the output must be less than $2^{22}$

</div>

> <font size="3">requires_grad()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">requires_grad_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">reshape()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">reshape_as()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">resize_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- `memory_format` only supports torch.contiguous_format and torch.preserve_format

</div>

> <font size="3">resize_as_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- `memory_format` only supports torch.contiguous_format and torch.preserve_format

</div>

> <font size="3">retain_grad()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">retains_grad()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">roll()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int32, int64, bool

</div>

> <font size="3">rot90()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">round()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">round_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">rsqrt()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">rsqrt_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">scatter()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, int16, int32, bool
- For <term>Ascend 950DT</term>, due to hardware differences, when indexes are duplicated, precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>.

</div>

> <font size="3">scatter_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- The `tensor`, `index`, and `src` parameters cannot be empty or scalars
- May fall back to CPU execution
- For <term>Ascend 950DT</term>, due to hardware differences, when indexes are duplicated, precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>.

</div>

> <font size="3">scatter_add_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- For <term>Ascend 950DT</term>, due to hardware differences, when indexes are duplicated, precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>.

</div>

> <font size="3">scatter_add()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp32
- For <term>Ascend 950DT</term>, due to hardware differences, when indexes are duplicated, precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>.

</div>

> <font size="3">scatter_reduce()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp32, int64
- May fall back to CPU execution

</div>

> <font size="3">select()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">select_scatter()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool
- May fall back to CPU execution

</div>

> <font size="3">set_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">share_memory_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">short()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">sigmoid_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">sign()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int32, int64, bool

</div>

> <font size="3">sign_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int32, int64, bool

</div>

> <font size="3">sgn()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int32, int64, bool, complex64, complex128

</div>

> <font size="3">sgn_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, int32, int64, bool

</div>

> <font size="3">sin()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">sin_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, complex64, complex128

</div>

> <font size="3">sinh()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, fp64

</div>

> <font size="3">sinh_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, fp64

</div>

> <font size="3">asinh()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">asinh_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">arcsinh()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">arcsinh_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">shape()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">size()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">slogdet()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32, complex64, complex128

</div>

> <font size="3">slice_scatter()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">softmax()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, fp64

</div>

> <font size="3">sort()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64
- For <term>Ascend 950DT</term>, due to underlying implementation limitations, "stable" only supports True. If it is set to False, it will be automatically modified to True at execution time.

</div>

> <font size="3">split()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">sparse_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">sparse_dim()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">sqrt()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">sqrt_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">square()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">square_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128

</div>

> <font size="3">squeeze()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">squeeze_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">std()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32
- `input` does not support scalar `tensor`
- The value of the `correction` parameter cannot exceed the maximum value of int32

</div>

> <font size="3">storage()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">untyped_storage()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">storage_offset()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">storage_type()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">stride()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">sub()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">sub_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128

</div>

> <font size="3">subtract_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">sum()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int32

</div>

> <font size="3">sum_to_size()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">swapaxes()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">swapdims()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">t()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128

</div>

> <font size="3">t_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64

</div>

> <font size="3">tensor_split()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: CPU-only

</div>

> <font size="3">tile()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- If the length of the input `dims` is smaller than the length of `Tensor`.shape, 1s are automatically prepended to `dims` to align its length with `Tensor`.shape. The filled `dims` must meet the following restrictions:
  - When the first axis needs to be repeated, a maximum of 4 dimensions can be repeated at the same time (that is, the number of elements greater than 1 in `dims` ≤ 4). For example, `Tensor`.tile([2, 3, 4, 5, 6]) is not supported, while `Tensor`.tile([2, 3, 1, 5, 6]) is supported.
  - When the first axis does not need to be repeated, a maximum of 3 dimensions can be repeated at the same time (that is, the number of elements greater than 1 in `dims` ≤ 3). For example, `Tensor`.tile([1, 3, 4, 5, 6]) is not supported, while `Tensor`.tile([1, 3, 1, 5, 6]) is supported.
  - In backward computation, the number of dimensions of `Tensor` plus the number of elements greater than 1 in `dims` must not exceed 8.

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- Current NPU devices only support setting `memory_format` to torch.contiguous_format or torch.preserve_format
- <term>Atlas inference products</term> do not support cross-NPU copy

</div>

> <font size="3">to_mkldnn()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">take()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int16, int32, bool

</div>

> <font size="3">take_along_dim()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int16, int32, int64, bool

</div>

> <font size="3">tan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128, with a value range of [-65504, 65504]

</div>

> <font size="3">tan_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">tanh()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64
- <term>Ascend 950DT</term>: Not Supported (fp64)

</div>

> <font size="3">tanh_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">atanh()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">atanh_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">arctanh()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">arctanh_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">tolist()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">topk()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64
- Due to hardware differences, the npu `topk` index results are inconsistent with those of GPU/CPU. Current NPUs only support returning computation results with `sorted` set to true.
- Scalar `tensor` is not supported.

</div>

> <font size="3">to_dense()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">to_sparse()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">to_sparse_csr()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">to_sparse_csc()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">to_sparse_bsr()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">to_sparse_bsc()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">transpose()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">transpose_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">tril()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">tril_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">triu()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">triu_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">true_divide()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">true_divide_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">trunc()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">trunc_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">type_as()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">unbind()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">unflatten()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">unfold()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">uniform_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64
- Following the PyTorch community convention, processing bool-type data is no longer supported. For existing bool-type data, use the following replacement methods: if all True values are required, use Tensor.bernoulli_(p=1.0). If uniformly distributed bool values are required, use Tensor.bernoulli_(p=0.5).

</div>

> <font size="3">unique()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- When the input contains 0, the output may contain positive 0 and negative 0 instead of only one 0

</div>

> <font size="3">unique_consecutive()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">unsqueeze()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">unsqueeze_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">values()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: Depends on the sparse `tensor`

</div>

> <font size="3">var()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32
- The value of the `correction` parameter cannot exceed the maximum value of int32

</div>

> <font size="3">view()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">view_as()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">vsplit()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">where()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">xlogy()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">xlogy_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">zero_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

</div>
