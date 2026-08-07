# torch.distributed.checkpoint

> [!NOTE]
>
> - API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。

## 目录

- [base API](#base-api)
- [Additional resources](#additional-resources)

## base API

### torch.distributed.checkpoint.state_dict_saver.save

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.state_dict_saver.save](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.distributed.checkpoint.state_dict_saver.save_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.state_dict_saver.save_state_dict](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.distributed.checkpoint.state_dict_loader.load_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.state_dict_loader.load_state_dict](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_loader.load_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.stateful.Stateful

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.stateful.Stateful](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.stateful.Stateful)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.stateful.Stateful.load_state_dict](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.stateful.Stateful.load_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.stateful.Stateful.state_dict](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.stateful.Stateful.state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.FileSystemReader

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.FileSystemReader](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemReader)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.FileSystemWriter

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.FileSystemWriter](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemWriter)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.staging.BlockingAsyncStager

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.staging.BlockingAsyncStager](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.staging.BlockingAsyncStager)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

> <font size="3">stage()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.staging.BlockingAsyncStager.stage](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.staging.BlockingAsyncStager.stage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">synchronize_staging()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.staging.BlockingAsyncStager.synchronize_staging](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.staging.BlockingAsyncStager.synchronize_staging)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.DefaultSavePlanner

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.DefaultSavePlanner](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultSavePlanner)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

> <font size="3">lookup_object()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.DefaultSavePlanner.lookup_object](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultSavePlanner.lookup_object)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">transform_object()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.DefaultSavePlanner.transform_object](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultSavePlanner.transform_object)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.DefaultLoadPlanner

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.DefaultLoadPlanner](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultLoadPlanner)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

> <font size="3">lookup_tensor()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.DefaultLoadPlanner.lookup_tensor](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultLoadPlanner.lookup_tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">transform_tensor()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.DefaultLoadPlanner.transform_tensor](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultLoadPlanner.transform_tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

</div>

### torch.distributed.checkpoint.state_dict.get_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.state_dict.get_state_dict](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.distributed.checkpoint.state_dict.get_model_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.state_dict.get_model_state_dict](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_model_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.distributed.checkpoint.state_dict.get_optimizer_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.state_dict.get_optimizer_state_dict](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_optimizer_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.distributed.checkpoint.state_dict.set_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.state_dict.set_state_dict](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.distributed.checkpoint.state_dict.set_model_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.state_dict.set_model_state_dict](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_model_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.distributed.checkpoint.state_dict.set_optimizer_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.state_dict.set_optimizer_state_dict](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_optimizer_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

> <font size="3">read_metadata()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_metadata](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_metadata)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">prepare_local_plan()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_local_plan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_local_plan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">prepare_global_plan()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_global_plan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_global_plan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">read_data()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_data](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_data)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.reset](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.reset)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">set_up_storage_reader()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.set_up_storage_reader](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.set_up_storage_reader)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">validate_checkpoint_id()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.validate_checkpoint_id](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.validate_checkpoint_id)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

## Additional resources

### <code><i>class</i></code> torch.distributed.checkpoint.StorageReader

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageReader](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

> <font size="3">prepare_global_plan()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageReader.prepare_global_plan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.prepare_global_plan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">prepare_local_plan()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageReader.prepare_local_plan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.prepare_local_plan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">read_data()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageReader.read_data](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.read_data)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">read_metadata()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageReader.read_metadata](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.read_metadata)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageReader.reset](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.reset)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">set_up_storage_reader()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageReader.set_up_storage_reader](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.set_up_storage_reader)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">validate_checkpoint_id()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageReader.validate_checkpoint_id](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.validate_checkpoint_id)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.StorageWriter

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageWriter](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

> <font size="3">finish()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageWriter.finish](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.finish)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">prepare_global_plan()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageWriter.prepare_global_plan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.prepare_global_plan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">prepare_local_plan()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageWriter.prepare_local_plan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.prepare_local_plan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageWriter.reset](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.reset)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">set_up_storage_writer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageWriter.set_up_storage_writer](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.set_up_storage_writer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">storage_meta()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageWriter.storage_meta](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.storage_meta)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">validate_checkpoint_id()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageWriter.validate_checkpoint_id](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.validate_checkpoint_id)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">write_data()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.StorageWriter.write_data](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.write_data)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.LoadPlanner

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.LoadPlanner](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

> <font size="3">commit_tensor()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.LoadPlanner.commit_tensor](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.commit_tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">create_global_plan()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.LoadPlanner.create_global_plan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.create_global_plan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">create_local_plan()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.LoadPlanner.create_local_plan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.create_local_plan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">finish_plan()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.LoadPlanner.finish_plan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.finish_plan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">load_bytes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.LoadPlanner.load_bytes](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.load_bytes)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">resolve_tensor()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.LoadPlanner.resolve_tensor](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.resolve_tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">set_up_planner()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.LoadPlanner.set_up_planner](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.set_up_planner)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.LoadPlan

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.LoadPlan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.ReadItem

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.ReadItem](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.ReadItem)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.SavePlanner

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.SavePlanner](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

> <font size="3">create_global_plan()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.SavePlanner.create_global_plan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.create_global_plan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">create_local_plan()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.SavePlanner.create_local_plan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.create_local_plan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">finish_plan()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.SavePlanner.finish_plan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.finish_plan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">resolve_data()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.SavePlanner.resolve_data](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.resolve_data)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">set_up_planner()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.SavePlanner.set_up_planner](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.set_up_planner)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.SavePlan

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.SavePlan](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlan)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.planner.WriteItem

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.planner.WriteItem](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.planner.WriteItem)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

> <font size="3">tensor_storage_size()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.planner.WriteItem.tensor_storage_size](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.planner.WriteItem.tensor_storage_size)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.staging.AsyncStager

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.staging.AsyncStager](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.staging.AsyncStager)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

> <font size="3">should_synchronize_after_execute()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.staging.AsyncStager.should_synchronize_after_execute](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.staging.AsyncStager.should_synchronize_after_execute)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">stage()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.staging.AsyncStager.stage](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.staging.AsyncStager.stage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">synchronize_staging()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.staging.AsyncStager.synchronize_staging](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.staging.AsyncStager.synchronize_staging)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### <code><i>class</i></code> torch.distributed.checkpoint.state_dict.StateDictOptions

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.checkpoint.state_dict.StateDictOptions](https://pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.StateDictOptions)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>
