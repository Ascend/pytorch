# tlparse / TORCH_TRACE

tlparse / `TORCH_TRACE`是一组生成[此类](https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html)编译报告的工具。

收集trace非常简单。按如下方式运行模型即可：

```bash
TORCH_TRACE="/tmp/tracedir" python foo.py
pip install tlparse
tlparse /tmp/tracedir --latest
```

`--latest`标志会处理目录中最新的日志。也可以使用`tlparse <log_file>`处理特定日志文件。

默认情况下，输出保存在`tl_out`文件夹中。也可以使用`-o my_folder`指定输出文件夹。

即使运行分布式作业，此方法也同样有效，并会为每个rank提供一份trace。它会在浏览器中打开与上文生成内容类似的HTML。如果要报告一个没有独立复现用例的复杂问题，仍可以通过以下方式为PyTorch开发者提供很大帮助：

1. 附上`/tmp/tracedir`中生成的trace日志；或者
2. 附上一个**包含tlparse全部输出的zip文件**（例如`tl_out`中的所有文件）。请勿只附加`index.html`文件，因为其中仅包含输出文件目录，不包含实际输出。

> [!NOTICE]
>
> trace日志包含全部模型代码。如果所处理的模型包含敏感信息，请勿共享trace日志。trace日志**不包含**权重。

`tlparse`的输出主要面向PyTorch开发者，其日志格式便于上传和分享到GitHub。不过，即使不是PyTorch开发者，也可以从中提取有用信息。建议先阅读报告中的内嵌帮助文本，其中说明了报告内容。通过`tlparse`可以获得以下信息：

- 通过查看堆栈trie，了解编译了哪些模型代码。如果不熟悉正在编译的代码库，这一点尤其有用。
- 存在多少个图断裂/不同的编译区域？每次不同的编译都会显示为独立的彩色块，例如`[0/0]`。可能发生图断裂的frame为浅绿色，例如`[2/4]`。如果frame很多，就值得怀疑，可能发生了灾难性的图断裂，或者代码不适合`torch.compile`。
- 某个frame重新编译了多少次？大量重新编译会显示为`[10/0]`、`[10/1]`、`[10/2]`。如果某项内容反复重新编译，即使它不是问题根因，也非常可疑，值得调查。
- 是否发生编译错误？出错的frame显示为红色，例如`[0/1]`。
- 给定frame生成了哪些中间编译产物？例如，可以查看生成的高层FX图或Triton代码。
- 某个特定frame是否存在相关信息？可以在`compilation_metrics`中找到这些信息。

以下列出了一些文件名及其说明。根据具体程序，可能不会看到所有这些文件。

| 文件名 | 说明 |
| --- | --- |
| `dynamo_output_graph` | Dynamo前端图捕获的输出图 |
| `before_pre_grad_graph` | 执行任何Autograd前图pass之前的FX图 |
| `after_pre_grad_graph` | 执行所有Autograd前图pass之后的FX图 |
| `aot_autograd_cache_miss` / `aot_autograd_cache_hit` | `aot_autograd_cache`的缓存键，以及缓存命中或未命中状态 |
| `aot_inference_graph` | 不需要Autograd时（例如所有Tensor都不要求梯度）分解后的FX图 |
| `aot_joint_graph` | Autograd和分解后的联合前向-反向图 |
| `aot_forward_graph` | 对`aot_joint_graph`分区后的前向图 |
| `aot_backward_graph` | 对`aot_joint_graph`分区后的反向图 |
| `before_joint_graph` | 执行任何联合图pass之前的FX图 |
| `after_joint_graph` | 执行所有联合图pass之后的FX图 |
| `before_post_grad_graph` | 执行任何Autograd后图pass之前的FX图 |
| `inductor_post_grad_graph` | 执行所有Autograd后图pass之后的FX图 |
| `fx_graph_runnable` | 与`before_post_grad_graph`基本相同，但它是可运行的Python脚本；还包含torch配置和一些包装代码，可使用虚拟输入运行该图 |
| `inductor_output_code` | Inductor生成的代码 |
| `fx_graph_cache_miss` / `fx_graph_cache_hit` | FX图缓存的缓存键，以及缓存命中或未命中状态 |
| `dynamo_cpp_guards_str` | Dynamo的guard信息 |

## TORCH_LOGS

可以使用`TORCH_LOGS`环境变量有选择地启用`torch.compile`技术栈各部分的日志。实际上，`TORCH_LOGS`就是`tlparse`的日志来源。`TORCH_LOGS`环境变量格式如下：

```bash
TORCH_LOGS="<option1>,<option2>,..." python foo.py
```

也可以使用`torch._logging.set_logs`以编程方式设置日志选项：

```python
import logging
torch._logging.set_logs(graph_breaks=True, dynamic=logging.DEBUG)
```

最有用的选项包括：

- `graph_breaks`：记录用户代码中的图断裂位置及其原因
- `guards`：记录生成的guard
- `recompiles`：记录发生重编译的函数以及导致重编译的失败guard
- `dynamic`：记录动态形状相关日志
- `output_code`：记录Inductor生成的代码

其他一些有用的`TORCH_LOGS`选项包括：

| 选项 | 说明 |
| --- | --- |
| `+all` | 输出所有`torch.compile`组件的调试日志 |
| `+dynamo` | 输出TorchDynamo调试日志 |
| `+aot` | 输出AOTAutograd调试日志 |
| `+inductor` | 输出TorchInductor调试日志 |
| `dynamic` | 输出动态形状日志 |
| `graph_code` | 输出Dynamo生成的FX图Python代码 |
| `graph_sizes` | 输出Dynamo生成的FX图中的Tensor大小 |
| `trace_bytecode` | 输出Dynamo正在追踪的字节码指令，以及Dynamo跟踪的符号解释器堆栈 |
| `trace_source` | 输出Dynamo当前正在追踪的原始源代码行 |
| `bytecode` | 输出Dynamo生成的字节码 |
| `guards` | 输出生成的guard |
| `recompiles` | 输出重编译原因（仅输出第一个失败的guard检查） |
| `recompiles_verbose` | 输出发生重编译时所有失败的guard检查 |
| `aot_graphs` | 输出AOTAutograd生成的图 |
| `aot_joint_graphs` | 输出AOTAutograd生成的联合前向-反向图 |
| `output_code` | 输出Inductor生成的代码 |
| `kernel_code` | 按kernel输出Inductor生成的代码 |
| `schedule` | 输出Inductor调度日志 |
| `perf_hints` | 输出Inductor性能提示日志 |
| `fusion` | 输出Inductor融合日志 |

有关完整选项列表，请参考[torch.\_logging](https://pytorch.org/docs/stable/logging.html)和[torch.\_logging.set_logs](https://pytorch.org/docs/stable/generated/torch._logging.set_logs.html#torch._logging.set_logs)。

## tlparse与TORCH_LOGS

通常，遇到问题时建议先使用`tlparse`。`tlparse`非常适合调试大型模型，以及从高层了解模型的编译方式。另一方面，对于小型示例和细粒度调试细节，如果已经大致知道是哪个`torch.compile`组件导致问题，则更适合使用`TORCH_LOGS`。
