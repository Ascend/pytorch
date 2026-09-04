# torch.futures

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/futures.html)。

<div style="display:none;">

## &#8203;torch.futures

</div>

### <code><i>class</i></code> torch.futures.Future

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

> <font size="3">add_done_callback()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.add_done_callback](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.add_done_callback)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">done()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.done](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.done)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_exception()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.set_exception](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.set_exception)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_result()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.set_result](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.set_result)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">then()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.then](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.then)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">value()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.value](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.value)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.futures.Future.wait](https://pytorch.org/docs/2.13/futures.html#torch.futures.Future.wait)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### torch.futures.collect_all

<div style="margin-left: 2em">

**原生文档**：[torch.futures.collect_all](https://pytorch.org/docs/2.13/futures.html#torch.futures.collect_all)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.futures.wait_all

<div style="margin-left: 2em">

**原生文档**：[torch.futures.wait_all](https://pytorch.org/docs/2.13/futures.html#torch.futures.wait_all)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>
