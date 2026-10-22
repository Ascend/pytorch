# torch.monitor

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.13/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.13/monitor.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [API Reference](#api-reference)

</div>

<div style="display:none;">

## &#8203;torch.monitor

</div>

## API Reference

### <code><i>class</i></code> torch.monitor.Aggregation

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Aggregation](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Aggregation)

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

> <font size="3">name()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Aggregation.name](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Aggregation.name)

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

</div>

### <code><i>class</i></code> torch.monitor.Stat

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Stat](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Stat)

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

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Stat.\_\_init\_\_](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Stat.__init__)

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

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Stat.add](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Stat.add)

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

</div>

> <font size="3">count()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Stat.count](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Stat.count)

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

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Stat.get](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Stat.get)

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

> <font size="3">name()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Stat.name](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Stat.name)

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

</div>

### <code><i>class</i></code> torch.monitor.Event

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Event](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Event)

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

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Event.\_\_init\_\_](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Event.__init__)

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

</div>

> <font size="3">data()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Event.data](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Event.data)

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

</div>

> <font size="3">name()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Event.name](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Event.name)

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

</div>

> <font size="3">timestamp()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.Event.timestamp](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.Event.timestamp)

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

</div>

</div>

### <code><i>class</i></code> torch.monitor.EventHandlerHandle

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.EventHandlerHandle](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.EventHandlerHandle)

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

</div>

### torch.monitor.log_event

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.log_event](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.log_event)

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

</div>

### torch.monitor.register_event_handler

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.register_event_handler](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.register_event_handler)

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

</div>

### torch.monitor.unregister_event_handler

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.unregister_event_handler](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.unregister_event_handler)

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

### <code><i>class</i></code> torch.monitor.TensorboardEventHandler

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.TensorboardEventHandler](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.TensorboardEventHandler)

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

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.TensorboardEventHandler.\_\_init\_\_](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.TensorboardEventHandler.__init__)

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

</div>

</div>

### torch.monitor.data_value_t

<div style="margin-left: 2em">

**Native Documentation**: [torch.monitor.data_value_t](https://pytorch.org/docs/2.13/monitor.html#torch.monitor.data_value_t)

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
