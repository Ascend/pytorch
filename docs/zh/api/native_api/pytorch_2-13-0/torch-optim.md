# torch.optim

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/optim.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Base class](#base-class)
- [Algorithms](#algorithms)
- [How to adjust learning rate](#how-to-adjust-learning-rate)

</div>

<div style="display:none;">

## &#8203;torch.optim

</div>

## Base class

### <code><i>class</i></code> torch.optim.Optimizer

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Optimizer](https://pytorch.org/docs/2.13/optim.html#torch.optim.Optimizer)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT</term>：支持
<!-- end id3 -->

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[Optimizer.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.Optimizer.add_param_group.html)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id6 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[Optimizer.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.Optimizer.load_state_dict.html)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id9 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[Optimizer.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.Optimizer.state_dict.html)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id12 -->

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[Optimizer.step](https://pytorch.org/docs/2.13/generated/torch.optim.Optimizer.step.html)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id15 -->

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[Optimizer.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.Optimizer.zero_grad.html)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id18 -->

</div>

</div>

## Algorithms

### <code><i>class</i></code> torch.optim.Adadelta

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta](https://pytorch.org/docs/2.13/generated/torch.optim.Adadelta.html)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id21 -->

**限制与说明**：

- `params`仅支持bf16，fp16，fp32
- 优化器在启动`foreach`的情况下（`foreach=None`或`foreach=True`），当被优化的参数分组过多时由于`foreach`算子的特性会导致性能下降。这种情况建议设置为`foreach=False`，例如：

  ```python
  # 参数分组较多时，建议关闭foreach避免性能下降
  optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-3, foreach=False)
  ```

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.add_param_group)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id24 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id27 -->

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT</term>：支持
<!-- end id30 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT</term>：支持
<!-- end id33 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id36 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT</term>：支持
<!-- end id39 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id42 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id45 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.state_dict)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id48 -->

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.step](https://pytorch.org/docs/2.13/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.step)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id51 -->

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.zero_grad)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id54 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.Adagrad

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad](https://pytorch.org/docs/2.13/generated/torch.optim.Adagrad.html)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT</term>：支持
<!-- end id57 -->

**限制与说明**：

- `params`仅支持bf16，fp16，fp32
- 优化器在启动`foreach`的情况下（`foreach=None`或`foreach=True`），当被优化的参数分组过多时由于`foreach`算子的特性会导致性能下降。这种情况建议设置为`foreach=False`，例如：

  ```python
  # 参数分组较多时，建议关闭foreach避免性能下降
  optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3, foreach=False)
  ```

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.add_param_group)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id60 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id63 -->

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id66 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT</term>：支持
<!-- end id69 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT</term>：支持
<!-- end id72 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT</term>：支持
<!-- end id75 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT</term>：支持
<!-- end id78 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id81 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.state_dict)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id84 -->

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.step](https://pytorch.org/docs/2.13/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.step)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id87 -->

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.zero_grad)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id90 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.Adam

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam](https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT</term>：支持
<!-- end id93 -->

**限制与说明**：

- `params`仅支持bf16，fp16，fp32
- 优化器在启动`foreach`的情况下（`foreach=None`或`foreach=True`），当被优化的参数分组过多时由于`foreach`算子的特性会导致性能下降。这种情况建议设置为`foreach=False`，例如：

  ```python
  # 参数分组较多时，建议关闭foreach避免性能下降
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=False)
  ```

- 在某些情况下可能回退至CPU执行
- 优化器在启动`fused`的情况下（`fused=True`），仅支持Ascend 950DT

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html#torch.optim.Adam.add_param_group)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id96 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html#torch.optim.Adam.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id99 -->

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html#torch.optim.Adam.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id102 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html#torch.optim.Adam.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT</term>：支持
<!-- end id105 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html#torch.optim.Adam.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id108 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html#torch.optim.Adam.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id111 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html#torch.optim.Adam.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT</term>：支持
<!-- end id114 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html#torch.optim.Adam.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT</term>：支持
<!-- end id117 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html#torch.optim.Adam.state_dict)

**产品支持情况**：

<!-- npu="910b" id118 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id120 -->

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.step](https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html#torch.optim.Adam.step)

**产品支持情况**：

<!-- npu="910b" id121 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id123 -->

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.Adam.html#torch.optim.Adam.zero_grad)

**产品支持情况**：

<!-- npu="910b" id124 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id126 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.AdamW

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW](https://pytorch.org/docs/2.13/generated/torch.optim.AdamW.html)

**产品支持情况**：

<!-- npu="910b" id127 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT</term>：支持
<!-- end id129 -->

**限制与说明**：

- `params`仅支持bf16，fp16，fp32，complex64
- 优化器在启动`foreach`的情况下（`foreach=None`或`foreach=True`），当被优化的参数分组过多时由于`foreach`算子的特性会导致性能下降。这种情况建议设置为`foreach=False`，例如：

  ```python
  # 参数分组较多时，建议关闭foreach避免性能下降
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
  ```

- 优化器在启动`fused`的情况下（`fused=True`），暂不支持`grad_scale`和`found_inf`参数。对标`_single_tensor_adamw`实现，fp32与cpu/cuda一致，fp16和bf16采用升精度实现，与cpu/cuda不一致

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.AdamW.html#torch.optim.AdamW.add_param_group)

**产品支持情况**：

<!-- npu="910b" id130 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT</term>：支持
<!-- end id132 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.AdamW.html#torch.optim.AdamW.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id133 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT</term>：支持
<!-- end id135 -->

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id136 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id138 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id139 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id141 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id142 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id144 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id145 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id147 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id148 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT</term>：支持
<!-- end id150 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id151 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id153 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.AdamW.html#torch.optim.AdamW.state_dict)

**产品支持情况**：

<!-- npu="910b" id154 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT</term>：支持
<!-- end id156 -->

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.step](https://pytorch.org/docs/2.13/generated/torch.optim.AdamW.html#torch.optim.AdamW.step)

**产品支持情况**：

<!-- npu="910b" id157 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT</term>：支持
<!-- end id159 -->

**限制与说明**：`params`仅支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.AdamW.html#torch.optim.AdamW.zero_grad)

**产品支持情况**：

<!-- npu="910b" id160 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT</term>：支持
<!-- end id162 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

</div>

### <code><i>class</i></code> torch.optim.SparseAdam

<div style="margin-left: 2em">

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.add_param_group)

**产品支持情况**：

<!-- npu="910b" id163 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT</term>：支持
<!-- end id165 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id166 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT</term>：支持
<!-- end id168 -->

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id169 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id171 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id172 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id174 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id175 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id177 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id178 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id180 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id181 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id183 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id184 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id186 -->

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.step](https://pytorch.org/docs/2.13/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.step)

**产品支持情况**：

<!-- npu="910b" id187 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id189 -->

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.zero_grad)

**产品支持情况**：

<!-- npu="910b" id190 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id192 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.Adamax

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax](https://pytorch.org/docs/2.13/generated/torch.optim.Adamax.html)

**产品支持情况**：

<!-- npu="910b" id193 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT</term>：支持
<!-- end id195 -->

**限制与说明**：

- `params`仅支持bf16，fp16，fp32
- 优化器在启动`foreach`的情况下（`foreach=None`或`foreach=True`），当被优化的参数分组过多时由于`foreach`算子的特性会导致性能下降。这种情况建议设置为`foreach=False`，例如：

  ```python
  # 参数分组较多时，建议关闭foreach避免性能下降
  optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, foreach=False)
  ```

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.Adamax.html#torch.optim.Adamax.add_param_group)

**产品支持情况**：

<!-- npu="910b" id196 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT</term>：支持
<!-- end id198 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.Adamax.html#torch.optim.Adamax.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id199 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id201 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id202 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id204 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id205 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT</term>：支持
<!-- end id207 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id208 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT</term>：支持
<!-- end id210 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id211 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT</term>：支持
<!-- end id213 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id214 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id216 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id217 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id219 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.Adamax.html#torch.optim.Adamax.state_dict)

**产品支持情况**：

<!-- npu="910b" id220 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id222 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.step](https://pytorch.org/docs/2.13/generated/torch.optim.Adamax.html#torch.optim.Adamax.step)

**产品支持情况**：

<!-- npu="910b" id223 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id225 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.Adamax.html#torch.optim.Adamax.zero_grad)

**产品支持情况**：

<!-- npu="910b" id226 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id228 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

</div>

### <code><i>class</i></code> torch.optim.ASGD

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD](https://pytorch.org/docs/2.13/generated/torch.optim.ASGD.html)

**产品支持情况**：

<!-- npu="910b" id229 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id231 -->

**限制与说明**： `params`仅支持fp16，fp32

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.ASGD.html#torch.optim.ASGD.add_param_group)

**产品支持情况**：

<!-- npu="910b" id232 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT</term>：支持
<!-- end id234 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.ASGD.html#torch.optim.ASGD.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id235 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT</term>：支持
<!-- end id237 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id238 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id238 -->
<!-- npu="A3" id239 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id239 -->
<!-- npu="950" id240 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id240 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id241 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id241 -->
<!-- npu="A3" id242 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id242 -->
<!-- npu="950" id243 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id243 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id244 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id244 -->
<!-- npu="A3" id245 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id245 -->
<!-- npu="950" id246 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id246 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id247 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id247 -->
<!-- npu="A3" id248 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id248 -->
<!-- npu="950" id249 -->
- <term>Ascend 950DT</term>：支持
<!-- end id249 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id250 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id250 -->
<!-- npu="A3" id251 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id251 -->
<!-- npu="950" id252 -->
- <term>Ascend 950DT</term>：支持
<!-- end id252 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id253 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id253 -->
<!-- npu="A3" id254 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id254 -->
<!-- npu="950" id255 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id255 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.ASGD.html#torch.optim.ASGD.state_dict)

**产品支持情况**：

<!-- npu="910b" id256 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id256 -->
<!-- npu="A3" id257 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id257 -->
<!-- npu="950" id258 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id258 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.step](https://pytorch.org/docs/2.13/generated/torch.optim.ASGD.html#torch.optim.ASGD.step)

**产品支持情况**：

<!-- npu="910b" id259 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id259 -->
<!-- npu="A3" id260 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id260 -->
<!-- npu="950" id261 -->
- <term>Ascend 950DT</term>：支持
<!-- end id261 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.ASGD.html#torch.optim.ASGD.zero_grad)

**产品支持情况**：

<!-- npu="910b" id262 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id262 -->
<!-- npu="A3" id263 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id263 -->
<!-- npu="950" id264 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id264 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

</div>

### <code><i>class</i></code> torch.optim.LBFGS

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS](https://pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html)

**产品支持情况**：

<!-- npu="910b" id265 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id265 -->
<!-- npu="A3" id266 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id266 -->
<!-- npu="950" id267 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id267 -->

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.add_param_group)

**产品支持情况**：

<!-- npu="910b" id268 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id268 -->
<!-- npu="A3" id269 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id269 -->
<!-- npu="950" id270 -->
- <term>Ascend 950DT</term>：支持
<!-- end id270 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id271 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id271 -->
<!-- npu="A3" id272 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id272 -->
<!-- npu="950" id273 -->
- <term>Ascend 950DT</term>：支持
<!-- end id273 -->

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id274 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id274 -->
<!-- npu="A3" id275 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id275 -->
<!-- npu="950" id276 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id276 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id277 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id277 -->
<!-- npu="A3" id278 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id278 -->
<!-- npu="950" id279 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id279 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id280 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id280 -->
<!-- npu="A3" id281 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id281 -->
<!-- npu="950" id282 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id282 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id283 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id283 -->
<!-- npu="A3" id284 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id284 -->
<!-- npu="950" id285 -->
- <term>Ascend 950DT</term>：支持
<!-- end id285 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id286 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id286 -->
<!-- npu="A3" id287 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id287 -->
<!-- npu="950" id288 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id288 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id289 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id289 -->
<!-- npu="A3" id290 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id290 -->
<!-- npu="950" id291 -->
- <term>Ascend 950DT</term>：支持
<!-- end id291 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.state_dict)

**产品支持情况**：

<!-- npu="910b" id292 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id292 -->
<!-- npu="A3" id293 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id293 -->
<!-- npu="950" id294 -->
- <term>Ascend 950DT</term>：支持
<!-- end id294 -->

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.step](https://pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.step)

**产品支持情况**：

<!-- npu="910b" id295 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id295 -->
<!-- npu="A3" id296 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id296 -->
<!-- npu="950" id297 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id297 -->

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.zero_grad)

**产品支持情况**：

<!-- npu="910b" id298 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id298 -->
<!-- npu="A3" id299 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id299 -->
<!-- npu="950" id300 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id300 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.NAdam

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam](https://pytorch.org/docs/2.13/generated/torch.optim.NAdam.html)

**产品支持情况**：

<!-- npu="910b" id301 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id301 -->
<!-- npu="A3" id302 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id302 -->
<!-- npu="950" id303 -->
- <term>Ascend 950DT</term>：支持
<!-- end id303 -->

**限制与说明**：

- `params`仅支持bf16，fp16，fp32
- 优化器在启动`foreach`的情况下（`foreach=None`或`foreach=True`），当被优化的参数分组过多时由于`foreach`算子的特性会导致性能下降。这种情况建议设置为`foreach=False`，例如：

  ```python
  # 参数分组较多时，建议关闭foreach避免性能下降
  optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3, foreach=False)
  ```

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.NAdam.html#torch.optim.NAdam.add_param_group)

**产品支持情况**：

<!-- npu="910b" id304 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id304 -->
<!-- npu="A3" id305 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id305 -->
<!-- npu="950" id306 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id306 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.NAdam.html#torch.optim.NAdam.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id307 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id307 -->
<!-- npu="A3" id308 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id308 -->
<!-- npu="950" id309 -->
- <term>Ascend 950DT</term>：支持
<!-- end id309 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id310 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id310 -->
<!-- npu="A3" id311 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id311 -->
<!-- npu="950" id312 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id312 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id313 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id313 -->
<!-- npu="A3" id314 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id314 -->
<!-- npu="950" id315 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id315 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id316 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id316 -->
<!-- npu="A3" id317 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id317 -->
<!-- npu="950" id318 -->
- <term>Ascend 950DT</term>：支持
<!-- end id318 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id319 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id319 -->
<!-- npu="A3" id320 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id320 -->
<!-- npu="950" id321 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id321 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id322 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id322 -->
<!-- npu="A3" id323 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id323 -->
<!-- npu="950" id324 -->
- <term>Ascend 950DT</term>：支持
<!-- end id324 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id325 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id325 -->
<!-- npu="A3" id326 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id326 -->
<!-- npu="950" id327 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id327 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.NAdam.html#torch.optim.NAdam.state_dict)

**产品支持情况**：

<!-- npu="910b" id328 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id328 -->
<!-- npu="A3" id329 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id329 -->
<!-- npu="950" id330 -->
- <term>Ascend 950DT</term>：支持
<!-- end id330 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.step](https://pytorch.org/docs/2.13/generated/torch.optim.NAdam.html#torch.optim.NAdam.step)

**产品支持情况**：

<!-- npu="910b" id331 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id331 -->
<!-- npu="A3" id332 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id332 -->
<!-- npu="950" id333 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id333 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.NAdam.html#torch.optim.NAdam.zero_grad)

**产品支持情况**：

<!-- npu="910b" id334 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id334 -->
<!-- npu="A3" id335 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id335 -->
<!-- npu="950" id336 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id336 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

</div>

### <code><i>class</i></code> torch.optim.RAdam

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam](https://pytorch.org/docs/2.13/generated/torch.optim.RAdam.html)

**产品支持情况**：

<!-- npu="910b" id337 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id337 -->
<!-- npu="A3" id338 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id338 -->
<!-- npu="950" id339 -->
- <term>Ascend 950DT</term>：支持
<!-- end id339 -->

**限制与说明**：

- `params`仅支持bf16，fp16，fp32
- 优化器在启动`foreach`的情况下（`foreach=None`或`foreach=True`），当被优化的参数分组过多时由于`foreach`算子的特性会导致性能下降。这种情况建议设置为`foreach=False`，例如：

  ```python
  # 参数分组较多时，建议关闭foreach避免性能下降
  optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, foreach=False)
  ```

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.RAdam.html#torch.optim.RAdam.add_param_group)

**产品支持情况**：

<!-- npu="910b" id340 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id340 -->
<!-- npu="A3" id341 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id341 -->
<!-- npu="950" id342 -->
- <term>Ascend 950DT</term>：支持
<!-- end id342 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.RAdam.html#torch.optim.RAdam.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id343 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id343 -->
<!-- npu="A3" id344 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id344 -->
<!-- npu="950" id345 -->
- <term>Ascend 950DT</term>：支持
<!-- end id345 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id346 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id346 -->
<!-- npu="A3" id347 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id347 -->
<!-- npu="950" id348 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id348 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id349 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id349 -->
<!-- npu="A3" id350 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id350 -->
<!-- npu="950" id351 -->
- <term>Ascend 950DT</term>：支持
<!-- end id351 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id352 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id352 -->
<!-- npu="A3" id353 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id353 -->
<!-- npu="950" id354 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id354 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id355 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id355 -->
<!-- npu="A3" id356 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id356 -->
<!-- npu="950" id357 -->
- <term>Ascend 950DT</term>：支持
<!-- end id357 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id358 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id358 -->
<!-- npu="A3" id359 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id359 -->
<!-- npu="950" id360 -->
- <term>Ascend 950DT</term>：支持
<!-- end id360 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id361 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id361 -->
<!-- npu="A3" id362 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id362 -->
<!-- npu="950" id363 -->
- <term>Ascend 950DT</term>：支持
<!-- end id363 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.RAdam.html#torch.optim.RAdam.state_dict)

**产品支持情况**：

<!-- npu="910b" id364 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id364 -->
<!-- npu="A3" id365 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id365 -->
<!-- npu="950" id366 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id366 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.step](https://pytorch.org/docs/2.13/generated/torch.optim.RAdam.html#torch.optim.RAdam.step)

**产品支持情况**：

<!-- npu="910b" id367 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id367 -->
<!-- npu="A3" id368 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id368 -->
<!-- npu="950" id369 -->
- <term>Ascend 950DT</term>：支持
<!-- end id369 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.RAdam.html#torch.optim.RAdam.zero_grad)

**产品支持情况**：

<!-- npu="910b" id370 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id370 -->
<!-- npu="A3" id371 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id371 -->
<!-- npu="950" id372 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id372 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

</div>

### <code><i>class</i></code> torch.optim.RMSprop

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop](https://pytorch.org/docs/2.13/generated/torch.optim.RMSprop.html)

**产品支持情况**：

<!-- npu="910b" id373 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id373 -->
<!-- npu="A3" id374 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id374 -->
<!-- npu="950" id375 -->
- <term>Ascend 950DT</term>：支持
<!-- end id375 -->

**限制与说明**：

- `params`仅支持bf16，fp16，fp32
- 优化器在启动`foreach`的情况下（`foreach=None`或`foreach=True`），当被优化的参数分组过多时由于`foreach`算子的特性会导致性能下降。这种情况建议设置为`foreach=False`，例如：

  ```python
  # 参数分组较多时，建议关闭foreach避免性能下降
  optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, foreach=False)
  ```

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.add_param_group)

**产品支持情况**：

<!-- npu="910b" id376 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id376 -->
<!-- npu="A3" id377 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id377 -->
<!-- npu="950" id378 -->
- <term>Ascend 950DT</term>：支持
<!-- end id378 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id379 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id379 -->
<!-- npu="A3" id380 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id380 -->
<!-- npu="950" id381 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id381 -->

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id382 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id382 -->
<!-- npu="A3" id383 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id383 -->
<!-- npu="950" id384 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id384 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id385 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id385 -->
<!-- npu="A3" id386 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id386 -->
<!-- npu="950" id387 -->
- <term>Ascend 950DT</term>：支持
<!-- end id387 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id388 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id388 -->
<!-- npu="A3" id389 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id389 -->
<!-- npu="950" id390 -->
- <term>Ascend 950DT</term>：支持
<!-- end id390 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id391 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id391 -->
<!-- npu="A3" id392 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id392 -->
<!-- npu="950" id393 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id393 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id394 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id394 -->
<!-- npu="A3" id395 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id395 -->
<!-- npu="950" id396 -->
- <term>Ascend 950DT</term>：支持
<!-- end id396 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id397 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id397 -->
<!-- npu="A3" id398 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id398 -->
<!-- npu="950" id399 -->
- <term>Ascend 950DT</term>：支持
<!-- end id399 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.state_dict)

**产品支持情况**：

<!-- npu="910b" id400 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id400 -->
<!-- npu="A3" id401 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id401 -->
<!-- npu="950" id402 -->
- <term>Ascend 950DT</term>：支持
<!-- end id402 -->

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.step](https://pytorch.org/docs/2.13/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.step)

**产品支持情况**：

<!-- npu="910b" id403 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id403 -->
<!-- npu="A3" id404 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id404 -->
<!-- npu="950" id405 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id405 -->

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.zero_grad)

**产品支持情况**：

<!-- npu="910b" id406 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id406 -->
<!-- npu="A3" id407 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id407 -->
<!-- npu="950" id408 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id408 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.Rprop

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop](https://pytorch.org/docs/2.13/generated/torch.optim.Rprop.html)

**产品支持情况**：

<!-- npu="910b" id409 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id409 -->
<!-- npu="A3" id410 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id410 -->
<!-- npu="950" id411 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id411 -->

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.Rprop.html#torch.optim.Rprop.add_param_group)

**产品支持情况**：

<!-- npu="910b" id412 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id412 -->
<!-- npu="A3" id413 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id413 -->
<!-- npu="950" id414 -->
- <term>Ascend 950DT</term>：支持
<!-- end id414 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.Rprop.html#torch.optim.Rprop.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id415 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id415 -->
<!-- npu="A3" id416 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id416 -->
<!-- npu="950" id417 -->
- <term>Ascend 950DT</term>：支持
<!-- end id417 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id418 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id418 -->
<!-- npu="A3" id419 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id419 -->
<!-- npu="950" id420 -->
- <term>Ascend 950DT</term>：支持
<!-- end id420 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id421 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id421 -->
<!-- npu="A3" id422 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id422 -->
<!-- npu="950" id423 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id423 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id424 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id424 -->
<!-- npu="A3" id425 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id425 -->
<!-- npu="950" id426 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id426 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id427 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id427 -->
<!-- npu="A3" id428 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id428 -->
<!-- npu="950" id429 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id429 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id430 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id430 -->
<!-- npu="A3" id431 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id431 -->
<!-- npu="950" id432 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id432 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id433 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id433 -->
<!-- npu="A3" id434 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id434 -->
<!-- npu="950" id435 -->
- <term>Ascend 950DT</term>：支持
<!-- end id435 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.Rprop.html#torch.optim.Rprop.state_dict)

**产品支持情况**：

<!-- npu="910b" id436 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id436 -->
<!-- npu="A3" id437 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id437 -->
<!-- npu="950" id438 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id438 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.step](https://pytorch.org/docs/2.13/generated/torch.optim.Rprop.html#torch.optim.Rprop.step)

**产品支持情况**：

<!-- npu="910b" id439 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id439 -->
<!-- npu="A3" id440 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id440 -->
<!-- npu="950" id441 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id441 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.Rprop.html#torch.optim.Rprop.zero_grad)

**产品支持情况**：

<!-- npu="910b" id442 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id442 -->
<!-- npu="A3" id443 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id443 -->
<!-- npu="950" id444 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id444 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

</div>

### <code><i>class</i></code> torch.optim.SGD

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD](https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html)

**产品支持情况**：

<!-- npu="910b" id445 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id445 -->
<!-- npu="A3" id446 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id446 -->
<!-- npu="950" id447 -->
- <term>Ascend 950DT</term>：支持
<!-- end id447 -->

**限制与说明**：

- `params`仅支持bf16，fp16，fp32
- 优化器在启动`foreach`的情况下（`foreach=None`或`foreach=True`），当被优化的参数分组过多时由于`foreach`算子的特性会导致性能下降。这种情况建议设置为`foreach=False`，例如：

  ```python
  # 参数分组较多时，建议关闭foreach避免性能下降
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, foreach=False)
  ```

- 优化器支持启动`fused`，即(`fused=True`)。

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.add_param_group](https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html#torch.optim.SGD.add_param_group)

**产品支持情况**：

<!-- npu="910b" id448 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id448 -->
<!-- npu="A3" id449 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id449 -->
<!-- npu="950" id450 -->
- <term>Ascend 950DT</term>：支持
<!-- end id450 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html#torch.optim.SGD.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id451 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id451 -->
<!-- npu="A3" id452 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id452 -->
<!-- npu="950" id453 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id453 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.register_load_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html#torch.optim.SGD.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id454 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id454 -->
<!-- npu="A3" id455 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id455 -->
<!-- npu="950" id456 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id456 -->

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html#torch.optim.SGD.register_load_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id457 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id457 -->
<!-- npu="A3" id458 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id458 -->
<!-- npu="950" id459 -->
- <term>Ascend 950DT</term>：支持
<!-- end id459 -->

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.register_state_dict_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html#torch.optim.SGD.register_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id460 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id460 -->
<!-- npu="A3" id461 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id461 -->
<!-- npu="950" id462 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id462 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.register_state_dict_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html#torch.optim.SGD.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id463 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id463 -->
<!-- npu="A3" id464 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id464 -->
<!-- npu="950" id465 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id465 -->

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.register_step_post_hook](https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html#torch.optim.SGD.register_step_post_hook)

**产品支持情况**：

<!-- npu="910b" id466 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id466 -->
<!-- npu="A3" id467 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id467 -->
<!-- npu="950" id468 -->
- <term>Ascend 950DT</term>：支持
<!-- end id468 -->

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.register_step_pre_hook](https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html#torch.optim.SGD.register_step_pre_hook)

**产品支持情况**：

<!-- npu="910b" id469 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id469 -->
<!-- npu="A3" id470 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id470 -->
<!-- npu="950" id471 -->
- <term>Ascend 950DT</term>：支持
<!-- end id471 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html#torch.optim.SGD.state_dict)

**产品支持情况**：

<!-- npu="910b" id472 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id472 -->
<!-- npu="A3" id473 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id473 -->
<!-- npu="950" id474 -->
- <term>Ascend 950DT</term>：支持
<!-- end id474 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.step](https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html#torch.optim.SGD.step)

**产品支持情况**：

<!-- npu="910b" id475 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id475 -->
<!-- npu="A3" id476 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id476 -->
<!-- npu="950" id477 -->
- <term>Ascend 950DT</term>：支持
<!-- end id477 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.zero_grad](https://pytorch.org/docs/2.13/generated/torch.optim.SGD.html#torch.optim.SGD.zero_grad)

**产品支持情况**：

<!-- npu="910b" id478 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id478 -->
<!-- npu="A3" id479 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id479 -->
<!-- npu="950" id480 -->
- <term>Ascend 950DT</term>：支持
<!-- end id480 -->

**限制与说明**： `params`仅支持fp16，fp32

</div>

</div>

## How to adjust learning rate

### <code><i>class</i></code> torch.optim.lr_scheduler.LambdaLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LambdaLR](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.LambdaLR.html)

**产品支持情况**：

<!-- npu="910b" id481 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id481 -->
<!-- npu="A3" id482 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id482 -->
<!-- npu="950" id483 -->
- <term>Ascend 950DT</term>：支持
<!-- end id483 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LambdaLR.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id484 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id484 -->
<!-- npu="A3" id485 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id485 -->
<!-- npu="950" id486 -->
- <term>Ascend 950DT</term>：支持
<!-- end id486 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LambdaLR.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id487 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id487 -->
<!-- npu="A3" id488 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id488 -->
<!-- npu="950" id489 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id489 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LambdaLR.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR.state_dict)

**产品支持情况**：

<!-- npu="910b" id490 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id490 -->
<!-- npu="A3" id491 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id491 -->
<!-- npu="950" id492 -->
- <term>Ascend 950DT</term>：支持
<!-- end id492 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.MultiplicativeLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiplicativeLR](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.MultiplicativeLR.html)

**产品支持情况**：

<!-- npu="910b" id493 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id493 -->
<!-- npu="A3" id494 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id494 -->
<!-- npu="950" id495 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id495 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiplicativeLR.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id496 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id496 -->
<!-- npu="A3" id497 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id497 -->
<!-- npu="950" id498 -->
- <term>Ascend 950DT</term>：支持
<!-- end id498 -->

**限制与说明**： `lr`仅支持fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiplicativeLR.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id499 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id499 -->
<!-- npu="A3" id500 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id500 -->
<!-- npu="950" id501 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id501 -->

**限制与说明**： `lr`仅支持fp32

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiplicativeLR.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR.state_dict)

**产品支持情况**：

<!-- npu="910b" id502 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id502 -->
<!-- npu="A3" id503 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id503 -->
<!-- npu="950" id504 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id504 -->

**限制与说明**： `lr`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.StepLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.StepLR](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.StepLR.html)

**产品支持情况**：

<!-- npu="910b" id505 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id505 -->
<!-- npu="A3" id506 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id506 -->
<!-- npu="950" id507 -->
- <term>Ascend 950DT</term>：支持
<!-- end id507 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.StepLR.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id508 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id508 -->
<!-- npu="A3" id509 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id509 -->
<!-- npu="950" id510 -->
- <term>Ascend 950DT</term>：支持
<!-- end id510 -->

**限制与说明**： `lr`仅支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.StepLR.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id511 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id511 -->
<!-- npu="A3" id512 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id512 -->
<!-- npu="950" id513 -->
- <term>Ascend 950DT</term>：支持
<!-- end id513 -->

**限制与说明**： `lr`仅支持fp16，fp32

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.StepLR.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR.state_dict)

**产品支持情况**：

<!-- npu="910b" id514 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id514 -->
<!-- npu="A3" id515 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id515 -->
<!-- npu="950" id516 -->
- <term>Ascend 950DT</term>：支持
<!-- end id516 -->

**限制与说明**： `lr`仅支持fp16，fp32

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.MultiStepLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiStepLR](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.MultiStepLR.html)

**产品支持情况**：

<!-- npu="910b" id517 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id517 -->
<!-- npu="A3" id518 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id518 -->
<!-- npu="950" id519 -->
- <term>Ascend 950DT</term>：支持
<!-- end id519 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiStepLR.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id520 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id520 -->
<!-- npu="A3" id521 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id521 -->
<!-- npu="950" id522 -->
- <term>Ascend 950DT</term>：支持
<!-- end id522 -->

**限制与说明**： `lr`仅支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiStepLR.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id523 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id523 -->
<!-- npu="A3" id524 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id524 -->
<!-- npu="950" id525 -->
- <term>Ascend 950DT</term>：支持
<!-- end id525 -->

**限制与说明**： `lr`仅支持fp16，fp32

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiStepLR.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR.state_dict)

**产品支持情况**：

<!-- npu="910b" id526 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id526 -->
<!-- npu="A3" id527 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id527 -->
<!-- npu="950" id528 -->
- <term>Ascend 950DT</term>：支持
<!-- end id528 -->

**限制与说明**： `lr`仅支持fp16，fp32

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.ConstantLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ConstantLR](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ConstantLR.html)

**产品支持情况**：

<!-- npu="910b" id529 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id529 -->
<!-- npu="A3" id530 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id530 -->
<!-- npu="950" id531 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id531 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ConstantLR.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id532 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id532 -->
<!-- npu="A3" id533 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id533 -->
<!-- npu="950" id534 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id534 -->

**限制与说明**： `lr`仅支持fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ConstantLR.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id535 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id535 -->
<!-- npu="A3" id536 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id536 -->
<!-- npu="950" id537 -->
- <term>Ascend 950DT</term>：支持
<!-- end id537 -->

**限制与说明**： `lr`仅支持fp32

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ConstantLR.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR.state_dict)

**产品支持情况**：

<!-- npu="910b" id538 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id538 -->
<!-- npu="A3" id539 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id539 -->
<!-- npu="950" id540 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id540 -->

**限制与说明**： `lr`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.LinearLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LinearLR](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.LinearLR.html)

**产品支持情况**：

<!-- npu="910b" id541 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id541 -->
<!-- npu="A3" id542 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id542 -->
<!-- npu="950" id543 -->
- <term>Ascend 950DT</term>：支持
<!-- end id543 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LinearLR.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id544 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id544 -->
<!-- npu="A3" id545 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id545 -->
<!-- npu="950" id546 -->
- <term>Ascend 950DT</term>：支持
<!-- end id546 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LinearLR.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id547 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id547 -->
<!-- npu="A3" id548 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id548 -->
<!-- npu="950" id549 -->
- <term>Ascend 950DT</term>：支持
<!-- end id549 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LinearLR.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR.state_dict)

**产品支持情况**：

<!-- npu="910b" id550 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id550 -->
<!-- npu="A3" id551 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id551 -->
<!-- npu="950" id552 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id552 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.ExponentialLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ExponentialLR](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ExponentialLR.html)

**产品支持情况**：

<!-- npu="910b" id553 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id553 -->
<!-- npu="A3" id554 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id554 -->
<!-- npu="950" id555 -->
- <term>Ascend 950DT</term>：支持
<!-- end id555 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ExponentialLR.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id556 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id556 -->
<!-- npu="A3" id557 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id557 -->
<!-- npu="950" id558 -->
- <term>Ascend 950DT</term>：支持
<!-- end id558 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ExponentialLR.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id559 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id559 -->
<!-- npu="A3" id560 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id560 -->
<!-- npu="950" id561 -->
- <term>Ascend 950DT</term>：支持
<!-- end id561 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ExponentialLR.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR.state_dict)

**产品支持情况**：

<!-- npu="910b" id562 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id562 -->
<!-- npu="A3" id563 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id563 -->
<!-- npu="950" id564 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id564 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.PolynomialLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.PolynomialLR](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.PolynomialLR.html)

**产品支持情况**：

<!-- npu="910b" id565 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id565 -->
<!-- npu="A3" id566 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id566 -->
<!-- npu="950" id567 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id567 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.PolynomialLR.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id568 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id568 -->
<!-- npu="A3" id569 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id569 -->
<!-- npu="950" id570 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id570 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.PolynomialLR.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id571 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id571 -->
<!-- npu="A3" id572 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id572 -->
<!-- npu="950" id573 -->
- <term>Ascend 950DT</term>：支持
<!-- end id573 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.PolynomialLR.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR.state_dict)

**产品支持情况**：

<!-- npu="910b" id574 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id574 -->
<!-- npu="A3" id575 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id575 -->
<!-- npu="950" id576 -->
- <term>Ascend 950DT</term>：支持
<!-- end id576 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.CosineAnnealingLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingLR](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)

**产品支持情况**：

<!-- npu="910b" id577 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id577 -->
<!-- npu="A3" id578 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id578 -->
<!-- npu="950" id579 -->
- <term>Ascend 950DT</term>：支持
<!-- end id579 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingLR.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id580 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id580 -->
<!-- npu="A3" id581 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id581 -->
<!-- npu="950" id582 -->
- <term>Ascend 950DT</term>：支持
<!-- end id582 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingLR.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id583 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id583 -->
<!-- npu="A3" id584 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id584 -->
<!-- npu="950" id585 -->
- <term>Ascend 950DT</term>：支持
<!-- end id585 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingLR.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR.state_dict)

**产品支持情况**：

<!-- npu="910b" id586 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id586 -->
<!-- npu="A3" id587 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id587 -->
<!-- npu="950" id588 -->
- <term>Ascend 950DT</term>：支持
<!-- end id588 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.ChainedScheduler

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ChainedScheduler](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ChainedScheduler.html)

**产品支持情况**：

<!-- npu="910b" id589 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id589 -->
<!-- npu="A3" id590 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id590 -->
<!-- npu="950" id591 -->
- <term>Ascend 950DT</term>：支持
<!-- end id591 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ChainedScheduler.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id592 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id592 -->
<!-- npu="A3" id593 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id593 -->
<!-- npu="950" id594 -->
- <term>Ascend 950DT</term>：支持
<!-- end id594 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ChainedScheduler.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id595 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id595 -->
<!-- npu="A3" id596 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id596 -->
<!-- npu="950" id597 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id597 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ChainedScheduler.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler.state_dict)

**产品支持情况**：

<!-- npu="910b" id598 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id598 -->
<!-- npu="A3" id599 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id599 -->
<!-- npu="950" id600 -->
- <term>Ascend 950DT</term>：支持
<!-- end id600 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.SequentialLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.SequentialLR](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.SequentialLR.html)

**产品支持情况**：

<!-- npu="910b" id601 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id601 -->
<!-- npu="A3" id602 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id602 -->
<!-- npu="950" id603 -->
- <term>Ascend 950DT</term>：支持
<!-- end id603 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.SequentialLR.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id604 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id604 -->
<!-- npu="A3" id605 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id605 -->
<!-- npu="950" id606 -->
- <term>Ascend 950DT</term>：支持
<!-- end id606 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.SequentialLR.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id607 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id607 -->
<!-- npu="A3" id608 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id608 -->
<!-- npu="950" id609 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id609 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.SequentialLR.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR.state_dict)

**产品支持情况**：

<!-- npu="910b" id610 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id610 -->
<!-- npu="A3" id611 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id611 -->
<!-- npu="950" id612 -->
- <term>Ascend 950DT</term>：支持
<!-- end id612 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.ReduceLROnPlateau

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ReduceLROnPlateau](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)

**产品支持情况**：

<!-- npu="910b" id613 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id613 -->
<!-- npu="A3" id614 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id614 -->
<!-- npu="950" id615 -->
- <term>Ascend 950DT</term>：支持
<!-- end id615 -->

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.CyclicLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CyclicLR](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CyclicLR.html)

**产品支持情况**：

<!-- npu="910b" id616 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id616 -->
<!-- npu="A3" id617 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id617 -->
<!-- npu="950" id618 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id618 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CyclicLR.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id619 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id619 -->
<!-- npu="A3" id620 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id620 -->
<!-- npu="950" id621 -->
- <term>Ascend 950DT</term>：支持
<!-- end id621 -->

</div>

> <font size="3">get_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CyclicLR.get_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR.get_lr)

**产品支持情况**：

<!-- npu="910b" id622 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id622 -->
<!-- npu="A3" id623 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id623 -->
<!-- npu="950" id624 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id624 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.OneCycleLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.OneCycleLR](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.OneCycleLR.html)

**产品支持情况**：

<!-- npu="910b" id625 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id625 -->
<!-- npu="A3" id626 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id626 -->
<!-- npu="950" id627 -->
- <term>Ascend 950DT</term>：支持
<!-- end id627 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.OneCycleLR.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id628 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id628 -->
<!-- npu="A3" id629 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id629 -->
<!-- npu="950" id630 -->
- <term>Ascend 950DT</term>：支持
<!-- end id630 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.OneCycleLR.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id631 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id631 -->
<!-- npu="A3" id632 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id632 -->
<!-- npu="950" id633 -->
- <term>Ascend 950DT</term>：支持
<!-- end id633 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.OneCycleLR.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR.state_dict)

**产品支持情况**：

<!-- npu="910b" id634 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id634 -->
<!-- npu="A3" id635 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id635 -->
<!-- npu="950" id636 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id636 -->

</div>

</div>

### <code><i>class</i></code> torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)

**产品支持情况**：

<!-- npu="910b" id637 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id637 -->
<!-- npu="A3" id638 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id638 -->
<!-- npu="950" id639 -->
- <term>Ascend 950DT</term>：支持
<!-- end id639 -->

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.get_last_lr](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.get_last_lr)

**产品支持情况**：

<!-- npu="910b" id640 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id640 -->
<!-- npu="A3" id641 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id641 -->
<!-- npu="950" id642 -->
- <term>Ascend 950DT</term>：支持
<!-- end id642 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.load_state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id643 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id643 -->
<!-- npu="A3" id644 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id644 -->
<!-- npu="950" id645 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id645 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.state_dict](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.state_dict)

**产品支持情况**：

<!-- npu="910b" id646 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id646 -->
<!-- npu="A3" id647 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id647 -->
<!-- npu="950" id648 -->
- <term>Ascend 950DT</term>：支持
<!-- end id648 -->

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.step](https://pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.step)

**产品支持情况**：

<!-- npu="910b" id649 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id649 -->
<!-- npu="A3" id650 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id650 -->
<!-- npu="950" id651 -->
- <term>Ascend 950DT</term>：支持
<!-- end id651 -->

</div>

</div>
