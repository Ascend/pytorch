# torch.fx

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/fx.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [API Reference](#api-reference)
- [torch.fx.node](#torchfxnode)
- [torch.fx.passes.regional_inductor](#torchfxpassesregional_inductor)

</div>

<div style="display:none;">

## &#8203;torch.fx

</div>

## API Reference

### torch.fx.symbolic_trace

<div style="margin-left: 2em">

**原生文档**：[torch.fx.symbolic_trace](https://pytorch.org/docs/2.13/fx.html#torch.fx.symbolic_trace)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.fx.GraphModule

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule](https://pytorch.org/docs/2.13/fx.html#torch.fx.GraphModule)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.\_\_init\_\_](https://pytorch.org/docs/2.13/fx.html#torch.fx.GraphModule.__init__)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">add_submodule()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.add_submodule](https://pytorch.org/docs/2.13/fx.html#torch.fx.GraphModule.add_submodule)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">code()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.code](https://pytorch.org/docs/2.13/fx.html#torch.fx.GraphModule.code)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">delete_all_unused_submodules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.delete_all_unused_submodules](https://pytorch.org/docs/2.13/fx.html#torch.fx.GraphModule.delete_all_unused_submodules)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">delete_submodule()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.delete_submodule](https://pytorch.org/docs/2.13/fx.html#torch.fx.GraphModule.delete_submodule)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">graph()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.graph](https://pytorch.org/docs/2.13/fx.html#torch.fx.GraphModule.graph)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">print_readable()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.print_readable](https://pytorch.org/docs/2.13/fx.html#torch.fx.GraphModule.print_readable)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">recompile()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.recompile](https://pytorch.org/docs/2.13/fx.html#torch.fx.GraphModule.recompile)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">to_folder()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.to_folder](https://pytorch.org/docs/2.13/fx.html#torch.fx.GraphModule.to_folder)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.fx.Graph

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.\_\_init\_\_](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.__init__)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.call_function](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.call_function)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">call_method()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.call_method](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.call_method)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.call_module](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.call_module)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">create_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.create_node](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.create_node)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">eliminate_dead_code()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.eliminate_dead_code](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.eliminate_dead_code)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">erase_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.erase_node](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.erase_node)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.get_attr](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.get_attr)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">graph_copy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.graph_copy](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.graph_copy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">find_nodes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.find_nodes](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.find_nodes)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">inserting_after()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.inserting_after](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.inserting_after)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">inserting_before()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.inserting_before](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.inserting_before)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">lint()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.lint](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.lint)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">node_copy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.node_copy](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.node_copy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">nodes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.nodes](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.nodes)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">on_generate_code()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.on_generate_code](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.on_generate_code)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">output()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.output](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.output)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">output_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.output_node](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.output_node)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">placeholder()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.placeholder](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.placeholder)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">print_tabular()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.print_tabular](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.print_tabular)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">process_inputs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.process_inputs](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.process_inputs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">process_outputs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.process_outputs](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.process_outputs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">python_code()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.python_code](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.python_code)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">set_codegen()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.set_codegen](https://pytorch.org/docs/2.13/fx.html#torch.fx.Graph.set_codegen)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.fx.Tracer

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.call_module](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.call_module)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">create_arg()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.create_arg](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.create_arg)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">create_args_for_root()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.create_args_for_root](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.create_args_for_root)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">create_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.create_node](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.create_node)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">create_proxy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.create_proxy](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.create_proxy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">getattr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.getattr](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.getattr)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_leaf_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.is_leaf_module](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.is_leaf_module)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">iter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.iter](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.iter)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">keys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.keys](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.keys)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">path_of_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.path_of_module](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.path_of_module)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">proxy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.proxy](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.proxy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">to_bool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.to_bool](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.to_bool)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">trace()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.trace](https://pytorch.org/docs/2.13/fx.html#torch.fx.Tracer.trace)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### torch.fx.wrap

<div style="margin-left: 2em">

**原生文档**：[torch.fx.wrap](https://pytorch.org/docs/2.13/fx.html#torch.fx.wrap)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

### torch.fx.graph.map_arg

<div style="margin-left: 2em">

**原生文档**：[torch.fx.graph.map_arg](https://pytorch.org/docs/2.13/fx.html#torch.fx.graph.map_arg)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.fx.Node

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

> <font size="3">all_input_nodes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.all_input_nodes](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.all_input_nodes)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">append()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.append](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.append)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.args](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.args)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">format_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.format_node](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.format_node)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">is_impure()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.is_impure](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.is_impure)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">kwargs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.kwargs](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.kwargs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">next()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.next](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.next)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">normalized_arguments()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.normalized_arguments](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.normalized_arguments)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">prepend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.prepend](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.prepend)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">prev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.prev](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.prev)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">replace_all_uses_with()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.replace_all_uses_with](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.replace_all_uses_with)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">replace_input_with()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.replace_input_with](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.replace_input_with)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">stack_trace()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.stack_trace](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.stack_trace)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">update_arg()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.update_arg](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.update_arg)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">update_kwarg()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.update_kwarg](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.update_kwarg)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">insert_arg()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.insert_arg](https://pytorch.org/docs/2.13/fx.html#torch.fx.Node.insert_arg)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.fx.Proxy

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Proxy](https://pytorch.org/docs/2.13/fx.html#torch.fx.Proxy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.fx.Interpreter

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.call_function](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter.call_function)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">call_method()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.call_method](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter.call_method)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.call_module](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter.call_module)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">fetch_args_kwargs_from_env()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.fetch_args_kwargs_from_env](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter.fetch_args_kwargs_from_env)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">fetch_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.fetch_attr](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter.fetch_attr)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.get_attr](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter.get_attr)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">map_nodes_to_values()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.map_nodes_to_values](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter.map_nodes_to_values)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">output()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.output](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter.output)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">placeholder()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.placeholder](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter.placeholder)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">run()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.run](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter.run)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">boxed_run()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.boxed_run](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter.boxed_run)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">run_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.run_node](https://pytorch.org/docs/2.13/fx.html#torch.fx.Interpreter.run_node)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.fx.Transformer

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Transformer](https://pytorch.org/docs/2.13/fx.html#torch.fx.Transformer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Transformer.call_function](https://pytorch.org/docs/2.13/fx.html#torch.fx.Transformer.call_function)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Transformer.call_module](https://pytorch.org/docs/2.13/fx.html#torch.fx.Transformer.call_module)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Transformer.get_attr](https://pytorch.org/docs/2.13/fx.html#torch.fx.Transformer.get_attr)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">placeholder()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Transformer.placeholder](https://pytorch.org/docs/2.13/fx.html#torch.fx.Transformer.placeholder)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">transform()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Transformer.transform](https://pytorch.org/docs/2.13/fx.html#torch.fx.Transformer.transform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### torch.fx.replace_pattern

<div style="margin-left: 2em">

**原生文档**：[torch.fx.replace_pattern](https://pytorch.org/docs/2.13/fx.html#torch.fx.replace_pattern)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.fx.node._type_repr

<div style="margin-left: 2em">

**原生文档**：[torch.fx.node._type_repr](https://pytorch.org/docs/2.13/fx.html#torch.fx.node._type_repr)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

## torch.fx.node

### torch.fx.node.map_arg

<div style="margin-left: 2em">

**原生文档**：[torch.fx.node.map_arg](https://pytorch.org/docs/2.13/fx.html#torch.fx.node.map_arg)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.fx.node.map_aggregate

<div style="margin-left: 2em">

**原生文档**：[torch.fx.node.map_aggregate](https://pytorch.org/docs/2.13/fx.html#torch.fx.node.map_aggregate)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

## torch.fx.passes.regional_inductor

### torch.fx.passes.regional_inductor.regional_inductor

<div style="margin-left: 2em">

**原生文档**：[torch.fx.passes.regional_inductor.regional_inductor](https://docs.pytorch.org/docs/2.13/generated/torch.fx.passes.regional_inductor.regional_inductor.html#torch.fx.passes.regional_inductor.regional_inductor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>
