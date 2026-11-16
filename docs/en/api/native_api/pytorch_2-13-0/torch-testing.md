# torch.testing

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.13/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.13/testing.html).

<div style="display:none;">

## &#8203;torch.testing

</div>

### torch.testing.assert_close

<div style="margin-left: 2em">

**Native Documentation**: [torch.testing.assert_close](https://pytorch.org/docs/2.13/testing.html#torch.testing.assert_close)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.testing.make_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.testing.make_tensor](https://pytorch.org/docs/2.13/testing.html#torch.testing.make_tensor)

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

**Constraints & Notes**: `input` only supports fp32, bool

</div>

<!-- npu="910b,A3" id9 -->
### torch.testing.assert_allclose

<div style="margin-left: 2em">

**Native Documentation**: [torch.testing.assert_allclose](https://pytorch.org/docs/2.13/testing.html#torch.testing.assert_allclose)

**Support Status**:
<!-- end id9 -->

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->

<!-- npu="910b,A3" id10 -->
**Constraints & Notes**: `input` only supports fp32

</div>
<!-- end id10 -->