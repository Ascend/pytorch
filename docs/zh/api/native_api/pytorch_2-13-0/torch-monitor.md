# torch.monitor

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/monitor.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [API Reference](#api-reference)

</div>

<div style="display:none;">

## &#8203;torch.monitor

</div>

## API Reference

### <code><i>class</i></code> torch.monitor.Aggregation

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Aggregation](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Aggregation)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Aggregation.name](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Aggregation.name)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.monitor.Stat

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Stat](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Stat)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Stat.\_\_init\_\_](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Stat.__init__)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Stat.add](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Stat.add)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">count()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Stat.count](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Stat.count)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Stat.get](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Stat.get)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Stat.name](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Stat.name)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.monitor.Event

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Event](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Event)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Event.\_\_init\_\_](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Event.__init__)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">data()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Event.data](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Event.data)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Event.name](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Event.name)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">timestamp()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.Event.timestamp](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Event.timestamp)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.monitor.EventHandlerHandle

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.EventHandlerHandle](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.EventHandlerHandle)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.monitor.log_event

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.log_event](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.log_event)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.monitor.register_event_handler

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.register_event_handler](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.register_event_handler)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.monitor.unregister_event_handler

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.unregister_event_handler](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.unregister_event_handler)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.monitor.TensorboardEventHandler

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.TensorboardEventHandler](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.TensorboardEventHandler)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.TensorboardEventHandler.\_\_init\_\_](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.TensorboardEventHandler.__init__)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### torch.monitor.data_value_t

<div style="margin-left: 2em">

**原生文档**：[torch.monitor.data_value_t](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.data_value_t)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>
