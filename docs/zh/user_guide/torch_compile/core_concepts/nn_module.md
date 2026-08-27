# PyTorch 2.0 NNModule支持

本文档内容与原生文档保持一致，原生社区文档详见[原生文档](https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_nn_module.html)。

`torch.compile`会对`torch.nn.Module`对象进行特殊处理，其追踪方式不同于任意Python类，目的是通过对结构作出假设来生成更快的代码。

本文介绍这种特化所带来的一些权衡和边界情况。

## NNModule Hook支持

早期`torch.compile`不支持nn.Module上的hook；即使注册，编译后的代码也会将其忽略。尽管许多用户仅在不使用hook或仅用于调试时才会涉及此功能，但将hook与`torch.compile`结合使用，在特定场景下仍具有合理性和必要性。

`torch.compile`目前对由`nn.Module.__call__`触发的 hook 提供部分支持，包括`_forward_pre_hooks`、`forward_hooks`、`_backward_pre_hooks`和`_backward_hooks`。将这类hook统称为“调用hook”。

而对于`_state_dict_hooks`及其`pre`和`load_`变体等与状态字典相关的 hook，`torch.compile`目前仍不支持。

## `nn.Module.__call__` Hook的用法

默认情况下，`torch.compile`会追踪`nn.Module.__call__`的内容，因此会遇到并运行forward/pre-forward hook。如果在调用`torch.compile`之前安装hook，之后不再删除或更改这些hook，则默认应支持此用例。

通常也支持backward/pre-backward hook，但有类似的注意事项：目前，访问backward_hooks字典时Dynamo会发生图断裂，这个问题或许可以通过进一步工作避免。图断裂还会影响backward hook的触发时机，因为图分段以Autograd函数的形式运行，并同时生成所有梯度。即使Dynamo能够在存在backward hook时不发生图断裂，一系列模块的backward hook仍会在整个编译图的反向传播运行完毕后一起触发。

**“允许模块”上的hook**：`torch.compile`会特殊处理`torch.conv`等常见模块以及难以追踪的模块，允许在Dynamo图中以不透明方式调用它们，而不由Dynamo追踪其内部。对于这些模块，hook当前会触发图断裂，使受影响的模块在Dynamo之外运行。根据模型的不同，这可能导致显著的性能下降，需要进一步工作来改进此项支持。

**skip_nnmodule_hook_guards**：默认情况下，`torch._dynamo.config.skip_nnmodule_hook_guards`设置为True，表示不会为每个nn.Module hook字典安装guard。这样可以减少guard执行时间、提高运行性能，但无法发现编译后hook字典发生的变化。

如果希望在编译后删除或修改hook，并让`torch.compile`作出适当响应（重新编译），则需要设置`skip_nnmodule_hook_guards=False`，同时接受额外guard带来的运行性能损失。

TODO：确认backward/pre_backward hook是否正常工作，并相应补充文档。

## state_dict Hook

`torch.compile`尚不支持state dict hook。

TODO：如果因hook发生图断裂，则使用warn_once；如果存在hook，则使用warn_once指向本文档。
