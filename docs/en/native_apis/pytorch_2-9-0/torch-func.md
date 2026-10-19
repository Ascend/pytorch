# torch.func

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://docs.pytorch.org/docs/2.9/func.api.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Function Transforms](#function-transforms)
- [Utilities for working with torch.nn.Modules](#utilities-for-working-with-torchnnmodules)

</div>

<div style="display:none;">

## &#8203;torch.func

</div>

## Function Transforms

### torch.func.grad_and_value

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.func.jvp

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.func.linearize

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.func.jacrev

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.func.jacfwd

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.func.hessian

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.func.functionalize

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.func.vmap

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.func.grad

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.func.vjp

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

## Utilities for working with torch.nn.Modules

### torch.func.functional_call

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.func.stack_module_state

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.func.replace_all_batch_norm_modules_

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>
