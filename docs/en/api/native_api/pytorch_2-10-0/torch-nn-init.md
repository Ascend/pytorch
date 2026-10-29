# torch.nn.init

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.10/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.10/nn.init.html).

<div style="display:none;">

## &#8203;torch.nn.init

</div>

### torch.nn.init.calculate_gain

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.calculate_gain](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.calculate_gain)

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

### torch.nn.init.uniform_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.uniform_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.uniform_)

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

**Constraints & Notes**: `tensor` only supports fp32

</div>

### torch.nn.init.normal_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.normal_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.normal_)

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

**Constraints & Notes**: `tensor` only supports fp32

</div>

### torch.nn.init.constant_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.constant_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.constant_)

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

**Constraints & Notes**: `tensor` only supports fp32

</div>

### torch.nn.init.ones_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.ones_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.ones_)

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

**Constraints & Notes**: `tensor` only supports fp32

</div>

### torch.nn.init.zeros_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.zeros_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.zeros_)

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

**Constraints & Notes**: `tensor` only supports fp32

</div>

### torch.nn.init.eye_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.eye_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.eye_)

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

**Constraints & Notes**: `tensor` only supports fp32

</div>

### torch.nn.init.dirac_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.dirac_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.dirac_)

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

**Constraints & Notes**: `tensor` only supports fp32

</div>

### torch.nn.init.xavier_uniform_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.xavier_uniform_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.xavier_uniform_)

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

**Constraints & Notes**: `tensor` only supports fp32

</div>

### torch.nn.init.xavier_normal_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.xavier_normal_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.xavier_normal_)

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

**Constraints & Notes**: `tensor` only supports fp32

</div>

### torch.nn.init.kaiming_uniform_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.kaiming_uniform_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.kaiming_uniform_)

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

**Constraints & Notes**: `tensor` only supports fp32

</div>

### torch.nn.init.kaiming_normal_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.kaiming_normal_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.kaiming_normal_)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

**Constraints & Notes**: `tensor` only supports fp32

</div>

### torch.nn.init.trunc_normal_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.trunc_normal_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.trunc_normal_)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id39 -->

</div>

### torch.nn.init.orthogonal_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.orthogonal_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.orthogonal_)

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

**Constraints & Notes**: `tensor` only supports fp32

</div>

### torch.nn.init.sparse_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.init.sparse_](https://pytorch.org/docs/2.10/nn.init.html#torch.nn.init.sparse_)

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

**Constraints & Notes**: `tensor` only supports fp32

</div>
