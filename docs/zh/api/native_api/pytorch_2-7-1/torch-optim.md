# torch.optim

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [How to use an optimizer](#how-to-use-an-optimizer)
- [Algorithms](#algorithms)
- [How to adjust learning rate](#how-to-adjust-learning-rate)

## How to use an optimizer

### _`class`_ torch.optim.Optimizer

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Optimizer](https://pytorch.org/docs/2.7/optim.html#torch.optim.Optimizer)

**是否支持**：是

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[Optimizer.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.Optimizer.add_param_group.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[Optimizer.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.Optimizer.load_state_dict.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[Optimizer.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.Optimizer.state_dict.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[Optimizer.step](https://pytorch.org/docs/2.7/generated/torch.optim.Optimizer.step.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[Optimizer.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.Optimizer.zero_grad.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## Algorithms

### _`class`_ torch.optim.Adadelta

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta](https://pytorch.org/docs/2.7/generated/torch.optim.Adadelta.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持bf16，fp16，fp32
- 优化器在启动foreach的情况下（foreach=None或foreach=True），当被优化的参数分组过多时由于foreach算子的特性会导致性能下降。这种情况建议设置为foreach=False

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.add_param_group)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_load_state_dict_post_hook)

**是否支持**：是

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_load_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_step_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_step_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.step](https://pytorch.org/docs/2.7/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.step)

**是否支持**：否

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adadelta.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.zero_grad)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.optim.Adagrad

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad](https://pytorch.org/docs/2.7/generated/torch.optim.Adagrad.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 优化器在启动foreach的情况下（foreach=None或foreach=True），当被优化的参数分组过多时由于foreach算子的特性会导致性能下降。这种情况建议设置为foreach=False

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.add_param_group)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_load_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_load_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_state_dict_post_hook)

**是否支持**：是

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_step_post_hook)

**是否支持**：是

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_step_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.step](https://pytorch.org/docs/2.7/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.step)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adagrad.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.zero_grad)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.optim.Adam

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam](https://pytorch.org/docs/2.7/generated/torch.optim.Adam.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 优化器在启动foreach的情况下（foreach=None或foreach=True），当被优化的参数分组过多时由于foreach算子的特性会导致性能下降。这种情况建议设置为foreach=False
- 在某些情况下可能回退至CPU执行

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.Adam.html#torch.optim.Adam.add_param_group)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.Adam.html#torch.optim.Adam.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adam.html#torch.optim.Adam.register_load_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adam.html#torch.optim.Adam.register_load_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adam.html#torch.optim.Adam.register_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adam.html#torch.optim.Adam.register_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adam.html#torch.optim.Adam.register_step_post_hook)

**是否支持**：是

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adam.html#torch.optim.Adam.register_step_pre_hook)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.Adam.html#torch.optim.Adam.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.step](https://pytorch.org/docs/2.7/generated/torch.optim.Adam.html#torch.optim.Adam.step)

**是否支持**：否

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adam.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.Adam.html#torch.optim.Adam.zero_grad)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.optim.AdamW

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW](https://pytorch.org/docs/2.7/generated/torch.optim.AdamW.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，complex64
- 优化器在启动foreach的情况下（foreach=None或foreach=True），当被优化的参数分组过多时由于foreach算子的特性会导致性能下降。这种情况建议设置为foreach=False
- 优化器在启动fused的情况下（fused=True），暂不支持grad_scale和found_inf参数。对标_single_tensor_adamw实现，fp32与cpu/cuda一致，fp16和bf16采用升精度实现，与cpu/cuda不一致

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.AdamW.html#torch.optim.AdamW.add_param_group)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.AdamW.html#torch.optim.AdamW.load_state_dict)

**是否支持**：是

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_load_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_load_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_step_post_hook)

**是否支持**：是

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_step_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.AdamW.html#torch.optim.AdamW.state_dict)

**是否支持**：是

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.step](https://pytorch.org/docs/2.7/generated/torch.optim.AdamW.html#torch.optim.AdamW.step)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.AdamW.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.AdamW.html#torch.optim.AdamW.zero_grad)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

</div>

### _`class`_ torch.optim.SparseAdam

<div style="margin-left: 2em">

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.add_param_group)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.load_state_dict)

**是否支持**：是

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_load_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_load_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_step_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_step_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.step](https://pytorch.org/docs/2.7/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.step)

**是否支持**：，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SparseAdam.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.zero_grad)

**是否支持**：，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.optim.Adamax

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax](https://pytorch.org/docs/2.7/generated/torch.optim.Adamax.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 优化器在启动foreach的情况下（oreach=None或foreach=True），当被优化的参数分组过多时由于foreach算子的特性会导致性能下降。这种情况建议设置为foreach=False

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.Adamax.html#torch.optim.Adamax.add_param_group)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.Adamax.html#torch.optim.Adamax.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_load_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_load_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_state_dict_post_hook)

**是否支持**：是

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_step_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_step_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.Adamax.html#torch.optim.Adamax.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.step](https://pytorch.org/docs/2.7/generated/torch.optim.Adamax.html#torch.optim.Adamax.step)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Adamax.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.Adamax.html#torch.optim.Adamax.zero_grad)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

</div>

### _`class`_ torch.optim.ASGD

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD](https://pytorch.org/docs/2.7/generated/torch.optim.ASGD.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.ASGD.html#torch.optim.ASGD.add_param_group)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.ASGD.html#torch.optim.ASGD.load_state_dict)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_load_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_load_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_step_post_hook)

**是否支持**：是

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_step_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.ASGD.html#torch.optim.ASGD.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.step](https://pytorch.org/docs/2.7/generated/torch.optim.ASGD.html#torch.optim.ASGD.step)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.ASGD.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.ASGD.html#torch.optim.ASGD.zero_grad)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

</div>

### _`class`_ torch.optim.LBFGS

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS](https://pytorch.org/docs/2.7/generated/torch.optim.LBFGS.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.add_param_group)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.load_state_dict)

**是否支持**：是

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_load_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_load_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_step_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_step_pre_hook)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.state_dict)

**是否支持**：是

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.step](https://pytorch.org/docs/2.7/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.step)

**是否支持**：否

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.LBFGS.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.zero_grad)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.optim.NAdam

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam](https://pytorch.org/docs/2.7/generated/torch.optim.NAdam.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 优化器在启动foreach的情况下（foreach=None或foreach=True），当被优化的参数分组过多时由于foreach算子的特性会导致性能下降。这种情况建议设置为foreach=False

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.NAdam.html#torch.optim.NAdam.add_param_group)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.NAdam.html#torch.optim.NAdam.load_state_dict)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_load_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_load_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_state_dict_post_hook)

**是否支持**：是

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_step_post_hook)

**是否支持**：是

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_step_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.NAdam.html#torch.optim.NAdam.state_dict)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.step](https://pytorch.org/docs/2.7/generated/torch.optim.NAdam.html#torch.optim.NAdam.step)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.NAdam.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.NAdam.html#torch.optim.NAdam.zero_grad)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

</div>

### _`class`_ torch.optim.RAdam

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam](https://pytorch.org/docs/2.7/generated/torch.optim.RAdam.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 优化器在启动foreach的情况下（foreach=None或foreach=True），当被优化的参数分组过多时由于foreach算子的特性会导致性能下降。这种情况建议设置为foreach=False

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.RAdam.html#torch.optim.RAdam.add_param_group)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.RAdam.html#torch.optim.RAdam.load_state_dict)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_load_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_load_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_step_post_hook)

**是否支持**：是

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_step_pre_hook)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.RAdam.html#torch.optim.RAdam.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.step](https://pytorch.org/docs/2.7/generated/torch.optim.RAdam.html#torch.optim.RAdam.step)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RAdam.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.RAdam.html#torch.optim.RAdam.zero_grad)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

</div>

### _`class`_ torch.optim.RMSprop

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop](https://pytorch.org/docs/2.7/generated/torch.optim.RMSprop.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 优化器在启动foreach的情况下（foreach=None或foreach=True），当被优化的参数分组过多时由于foreach算子的特性会导致性能下降。这种情况建议设置为foreach=False

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.add_param_group)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_load_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_load_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_state_dict_post_hook)

**是否支持**：是

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_step_post_hook)

**是否支持**：是

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_step_pre_hook)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.state_dict)

**是否支持**：是

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.step](https://pytorch.org/docs/2.7/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.step)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.RMSprop.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.zero_grad)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.optim.Rprop

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop](https://pytorch.org/docs/2.7/generated/torch.optim.Rprop.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.Rprop.html#torch.optim.Rprop.add_param_group)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.Rprop.html#torch.optim.Rprop.load_state_dict)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_load_state_dict_post_hook)

**是否支持**：是

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_load_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_step_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_step_pre_hook)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.Rprop.html#torch.optim.Rprop.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.step](https://pytorch.org/docs/2.7/generated/torch.optim.Rprop.html#torch.optim.Rprop.step)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.Rprop.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.Rprop.html#torch.optim.Rprop.zero_grad)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

</div>

### _`class`_ torch.optim.SGD

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD](https://pytorch.org/docs/2.7/generated/torch.optim.SGD.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 优化器在启动foreach的情况下（foreach=None或foreach=True），当被优化的参数分组过多时由于foreach算子的特性会导致性能下降。这种情况建议设置为foreach=False

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.add_param_group](https://pytorch.org/docs/2.7/generated/torch.optim.SGD.html#torch.optim.SGD.add_param_group)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.SGD.html#torch.optim.SGD.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.SGD.html#torch.optim.SGD.register_load_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.SGD.html#torch.optim.SGD.register_load_state_dict_pre_hook)

**是否支持**：是

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.SGD.html#torch.optim.SGD.register_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.SGD.html#torch.optim.SGD.register_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_step_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.register_step_post_hook](https://pytorch.org/docs/2.7/generated/torch.optim.SGD.html#torch.optim.SGD.register_step_post_hook)

**是否支持**：是

</div>

> <font size="3">register_step_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.register_step_pre_hook](https://pytorch.org/docs/2.7/generated/torch.optim.SGD.html#torch.optim.SGD.register_step_pre_hook)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.SGD.html#torch.optim.SGD.state_dict)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.step](https://pytorch.org/docs/2.7/generated/torch.optim.SGD.html#torch.optim.SGD.step)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.SGD.zero_grad](https://pytorch.org/docs/2.7/generated/torch.optim.SGD.html#torch.optim.SGD.zero_grad)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

</div>

## How to adjust learning rate

### _`class`_ torch.optim.lr_scheduler.LambdaLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LambdaLR](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.LambdaLR.html)

**是否支持**：是

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LambdaLR.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR.get_last_lr)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LambdaLR.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LambdaLR.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR.state_dict)

**是否支持**：是

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.MultiplicativeLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiplicativeLR](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.MultiplicativeLR.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiplicativeLR.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR.get_last_lr)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiplicativeLR.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiplicativeLR.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.StepLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.StepLR](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.StepLR.html)

**是否支持**：是

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.StepLR.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR.get_last_lr)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.StepLR.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR.load_state_dict)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.StepLR.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR.state_dict)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.MultiStepLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiStepLR](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.MultiStepLR.html)

**是否支持**：是

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiStepLR.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR.get_last_lr)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiStepLR.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR.load_state_dict)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.MultiStepLR.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR.state_dict)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.ConstantLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ConstantLR](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ConstantLR.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ConstantLR.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR.get_last_lr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ConstantLR.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR.load_state_dict)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ConstantLR.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.LinearLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LinearLR](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.LinearLR.html)

**是否支持**：是

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LinearLR.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR.get_last_lr)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LinearLR.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR.load_state_dict)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.LinearLR.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.ExponentialLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ExponentialLR](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ExponentialLR.html)

**是否支持**：是

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ExponentialLR.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR.get_last_lr)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ExponentialLR.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR.load_state_dict)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ExponentialLR.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.PolynomialLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.PolynomialLR](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.PolynomialLR.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.PolynomialLR.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR.get_last_lr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.PolynomialLR.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR.load_state_dict)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.PolynomialLR.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR.state_dict)

**是否支持**：是

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.CosineAnnealingLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingLR](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)

**是否支持**：是

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingLR.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR.get_last_lr)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingLR.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR.load_state_dict)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingLR.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR.state_dict)

**是否支持**：是

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.ChainedScheduler

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ChainedScheduler](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ChainedScheduler.html)

**是否支持**：是

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ChainedScheduler.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler.get_last_lr)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ChainedScheduler.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ChainedScheduler.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler.state_dict)

**是否支持**：是

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.SequentialLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.SequentialLR](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.SequentialLR.html)

**是否支持**：是

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.SequentialLR.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR.get_last_lr)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.SequentialLR.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.SequentialLR.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR.state_dict)

**是否支持**：是

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.ReduceLROnPlateau

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.ReduceLROnPlateau](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)

**是否支持**：是

</div>

### _`class`_ torch.optim.lr_scheduler.CyclicLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CyclicLR](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CyclicLR.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CyclicLR.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR.get_last_lr)

**是否支持**：是

</div>

> <font size="3">get_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CyclicLR.get_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR.get_lr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.OneCycleLR

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.OneCycleLR](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.OneCycleLR.html)

**是否支持**：是

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.OneCycleLR.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR.get_last_lr)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.OneCycleLR.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR.load_state_dict)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.OneCycleLR.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)

**是否支持**：是

> <font size="3">get_last_lr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.get_last_lr](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.get_last_lr)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.state_dict](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.state_dict)

**是否支持**：是

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.step](https://pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.step)

**是否支持**：是

</div>

</div>
