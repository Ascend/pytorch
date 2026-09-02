# 处理图断裂

正如您可能还记得的那样，[Dynamo核心概念](dynamo_core_concepts.md)中介绍过，Dynamo遇到无法追踪的代码时会产生图断裂。在`torch.compile`的默认设置下，Dynamo会编译截至该位置已确定的FX图，以常规Python方式执行不受支持的代码，然后恢复追踪。

借助图断裂，Dynamo可以追踪任意Python代码，并从中划分出可分别进行优化的函数式子图。

但是，图断裂可能导致`torch.compile`出现意外的性能下降。如果未获得预期的加速效果，建议检查并消除图断裂。

以下各节介绍处理图断裂的策略。

- [使用`fullgraph=True`识别并消除图断裂](fullgraph_true.md)
- [常见图断裂](common_graph_breaks.md)
- [使用`torch._dynamo.nonstrict_trace`](dynamo_nonstrict_trace.md)
- [自定义算子](custom_ops.md)
- [使用`fullgraph=False`](fullgraph_false.md)
