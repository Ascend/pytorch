# torch.distributed.checkpoint

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.9/distributed.checkpoint.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Additional resources](#additional-resources)

</div>

<div style="display:none;">

## &#8203;torch.distributed.checkpoint

</div>

## Additional resources

### torch.distributed.checkpoint.state_dict_saver.save

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.state_dict_saver.save](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save)

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

### torch.distributed.checkpoint.state_dict_saver.save_state_dict

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.state_dict_saver.save_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save_state_dict)

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

</div>

### torch.distributed.checkpoint.state_dict_loader.load_state_dict

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.state_dict_loader.load_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_loader.load_state_dict)

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

### <code><i>class</i></code> torch.distributed.checkpoint.stateful.Stateful

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.stateful.Stateful](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.stateful.Stateful)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id12 -->

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.stateful.Stateful.load_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.stateful.Stateful.load_state_dict)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id15 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.stateful.Stateful.state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.stateful.Stateful.state_dict)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id18 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.FileSystemReader

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.FileSystemReader](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemReader)

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

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.FileSystemWriter

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.FileSystemWriter](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemWriter)

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

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.staging.AsyncStager

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.staging.AsyncStager](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.AsyncStager)

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

> <font size="3">should_synchronize_after_execute()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.staging.AsyncStager.should_synchronize_after_execute](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.AsyncStager.should_synchronize_after_execute)

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

> <font size="3">stage()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.staging.AsyncStager.stage](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.AsyncStager.stage)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id33 -->

</div>

> <font size="3">synchronize_staging()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.staging.AsyncStager.synchronize_staging](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.AsyncStager.synchronize_staging)

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

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.staging.DefaultStager

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.staging.DefaultStager](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.DefaultStager)

**Support Status**:

<!-- npu="910b" id223 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT</term>: Not verified (no multi-card environment in A5)
<!-- end id225 -->

> <font size="3">close()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.staging.DefaultStager.close](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.DefaultStager.close)

**Support Status**:

<!-- npu="910b" id226 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT</term>: Not verified (no multi-card environment in A5)
<!-- end id228 -->

</div>

> <font size="3">stage()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.staging.DefaultStager.stage](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.DefaultStager.stage)

**Support Status**:

<!-- npu="910b" id229 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT</term>: Not verified (no multi-card environment in A5)
<!-- end id231 -->

</div>

> <font size="3">synchronize_staging()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.staging.DefaultStager.synchronize_staging](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.DefaultStager.synchronize_staging)

**Support Status**:

<!-- npu="910b" id232 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT</term>: Not verified (no multi-card environment in A5)
<!-- end id234 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.staging.StagingOptions

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.staging.StagingOptions](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.StagingOptions)

**Support Status**:

<!-- npu="910b" id235 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id237 -->

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.staging.BlockingAsyncStager

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.staging.BlockingAsyncStager](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.BlockingAsyncStager)

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

> <font size="3">stage()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.staging.BlockingAsyncStager.stage](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.BlockingAsyncStager.stage)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id42 -->

</div>

> <font size="3">synchronize_staging()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.staging.BlockingAsyncStager.synchronize_staging](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.BlockingAsyncStager.synchronize_staging)

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

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.DefaultSavePlanner

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.DefaultSavePlanner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultSavePlanner)

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

> <font size="3">lookup_object()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.DefaultSavePlanner.lookup_object](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultSavePlanner.lookup_object)

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

</div>

> <font size="3">transform_object()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.DefaultSavePlanner.transform_object](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultSavePlanner.transform_object)

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

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.DefaultLoadPlanner

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.DefaultLoadPlanner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultLoadPlanner)

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

> <font size="3">lookup_tensor()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.DefaultLoadPlanner.lookup_tensor](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultLoadPlanner.lookup_tensor)

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

</div>

> <font size="3">transform_tensor()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.DefaultLoadPlanner.transform_tensor](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultLoadPlanner.transform_tensor)

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

</div>

### torch.distributed.checkpoint.state_dict.get_state_dict

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.state_dict.get_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id66 -->

</div>

### torch.distributed.checkpoint.state_dict.get_model_state_dict

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.state_dict.get_model_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_model_state_dict)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id69 -->

</div>

### torch.distributed.checkpoint.state_dict.get_optimizer_state_dict

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.state_dict.get_optimizer_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_optimizer_state_dict)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id72 -->

</div>

### torch.distributed.checkpoint.state_dict.set_state_dict

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.state_dict.set_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_state_dict)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id75 -->

</div>

### torch.distributed.checkpoint.state_dict.set_model_state_dict

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.state_dict.set_model_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_model_state_dict)

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id78 -->

</div>

### torch.distributed.checkpoint.state_dict.set_optimizer_state_dict

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.state_dict.set_optimizer_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_optimizer_state_dict)

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id81 -->

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader)

**Support Status**:

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id84 -->

> <font size="3">read_metadata()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_metadata](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_metadata)

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id87 -->

</div>

> <font size="3">prepare_local_plan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_local_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_local_plan)

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id90 -->

</div>

> <font size="3">prepare_global_plan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_global_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_global_plan)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id93 -->

</div>

> <font size="3">read_data()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_data](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_data)

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

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.reset](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.reset)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id99 -->

</div>

> <font size="3">set_up_storage_reader()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.set_up_storage_reader](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.set_up_storage_reader)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id102 -->

</div>

> <font size="3">validate_checkpoint_id()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.validate_checkpoint_id](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.validate_checkpoint_id)

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id105 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner)

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id108 -->

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.StorageReader

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageReader](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader)

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

> <font size="3">prepare_global_plan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageReader.prepare_global_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.prepare_global_plan)

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

</div>

> <font size="3">prepare_local_plan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageReader.prepare_local_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.prepare_local_plan)

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

</div>

> <font size="3">read_data()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageReader.read_data](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.read_data)

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

</div>

> <font size="3">read_metadata()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageReader.read_metadata](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.read_metadata)

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

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageReader.reset](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.reset)

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

</div>

> <font size="3">set_up_storage_reader()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageReader.set_up_storage_reader](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.set_up_storage_reader)

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

</div>

> <font size="3">validate_checkpoint_id()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageReader.validate_checkpoint_id](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.validate_checkpoint_id)

**Support Status**:

<!-- npu="910b" id130 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id132 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.StorageWriter

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageWriter](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter)

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

> <font size="3">finish()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageWriter.finish](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.finish)

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

</div>

> <font size="3">prepare_global_plan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageWriter.prepare_global_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.prepare_global_plan)

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

</div>

> <font size="3">prepare_local_plan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageWriter.prepare_local_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.prepare_local_plan)

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

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageWriter.reset](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.reset)

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

</div>

> <font size="3">set_up_storage_writer()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageWriter.set_up_storage_writer](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.set_up_storage_writer)

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

</div>

> <font size="3">storage_meta()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageWriter.storage_meta](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.storage_meta)

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

</div>

> <font size="3">validate_checkpoint_id()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageWriter.validate_checkpoint_id](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.validate_checkpoint_id)

**Support Status**:

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id156 -->

</div>

> <font size="3">write_data()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.StorageWriter.write_data](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.write_data)

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

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.LoadPlanner

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.LoadPlanner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner)

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

> <font size="3">commit_tensor()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.LoadPlanner.commit_tensor](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.commit_tensor)

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

</div>

> <font size="3">create_global_plan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.LoadPlanner.create_global_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.create_global_plan)

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

</div>

> <font size="3">create_local_plan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.LoadPlanner.create_local_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.create_local_plan)

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

</div>

> <font size="3">finish_plan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.LoadPlanner.finish_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.finish_plan)

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

</div>

> <font size="3">load_bytes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.LoadPlanner.load_bytes](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.load_bytes)

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

</div>

> <font size="3">resolve_tensor()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.LoadPlanner.resolve_tensor](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.resolve_tensor)

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

</div>

> <font size="3">set_up_planner()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.LoadPlanner.set_up_planner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.set_up_planner)

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

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.LoadPlan

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.LoadPlan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlan)

**Support Status**:

<!-- npu="910b" id184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id186 -->

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.ReadItem

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.ReadItem](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.ReadItem)

**Support Status**:

<!-- npu="910b" id187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id189 -->

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.metadata.MetadataIndex

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.ReadItem](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.ReadItem)

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

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.SavePlanner

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.SavePlanner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner)

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

> <font size="3">create_global_plan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.SavePlanner.create_global_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.create_global_plan)

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

</div>

> <font size="3">create_local_plan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.SavePlanner.create_local_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.create_local_plan)

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

</div>

> <font size="3">finish_plan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.SavePlanner.finish_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.finish_plan)

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

</div>

> <font size="3">resolve_data()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.SavePlanner.resolve_data](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.resolve_data)

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

</div>

> <font size="3">set_up_planner()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.SavePlanner.set_up_planner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.set_up_planner)

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

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.SavePlan

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.SavePlan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlan)

**Support Status**:

<!-- npu="910b" id211 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id213 -->

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.planner.WriteItem

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.planner.WriteItem](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.planner.WriteItem)

**Support Status**:

<!-- npu="910b" id214 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id216 -->

> <font size="3">tensor_storage_size()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.planner.WriteItem.tensor_storage_size](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.planner.WriteItem.tensor_storage_size)

**Support Status**:

<!-- npu="910b" id217 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id219 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.state_dict.StateDictOptions

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.checkpoint.state_dict.StateDictOptions](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.StateDictOptions)

**Support Status**:

<!-- npu="910b" id220 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id222 -->

</div>
