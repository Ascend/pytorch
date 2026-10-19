# torch.distributed.checkpoint

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.11/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.11/distributed.checkpoint.html).

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

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.distributed.checkpoint.state_dict_saver.save_state_dict

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.distributed.checkpoint.state_dict_loader.load_state_dict

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.stateful.Stateful

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.FileSystemReader

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.FileSystemWriter

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.staging.BlockingAsyncStager

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">stage()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">synchronize_staging()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.DefaultSavePlanner

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

> <font size="3">lookup_object()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">transform_object()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.DefaultLoadPlanner

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

> <font size="3">lookup_tensor()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">transform_tensor()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### torch.distributed.checkpoint.state_dict.get_state_dict

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.distributed.checkpoint.state_dict.get_model_state_dict

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.distributed.checkpoint.state_dict.get_optimizer_state_dict

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.distributed.checkpoint.state_dict.set_state_dict

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.distributed.checkpoint.state_dict.set_model_state_dict

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.distributed.checkpoint.state_dict.set_optimizer_state_dict

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">read_metadata()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">prepare_local_plan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">prepare_global_plan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">read_data()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">set_up_storage_reader()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">validate_checkpoint_id()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.StorageReader

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

> <font size="3">prepare_global_plan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">prepare_local_plan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">read_data()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">read_metadata()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">set_up_storage_reader()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">validate_checkpoint_id()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.StorageWriter

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

> <font size="3">finish()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">prepare_global_plan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">prepare_local_plan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">set_up_storage_writer()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">storage_meta()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">validate_checkpoint_id()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">write_data()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.LoadPlanner

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

> <font size="3">commit_tensor()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">create_global_plan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">create_local_plan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">finish_plan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">load_bytes()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">resolve_tensor()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">set_up_planner()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.LoadPlan

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.ReadItem

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.SavePlanner

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

> <font size="3">create_global_plan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">create_local_plan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">finish_plan()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">resolve_data()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">set_up_planner()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.SavePlan

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.planner.WriteItem

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">tensor_storage_size()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.staging.AsyncStager

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">should_synchronize_after_execute()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">stage()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">synchronize_staging()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.state_dict.StateDictOptions

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>
