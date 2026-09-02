# 使用`fullgraph=False`

虽然`fullgraph=False`是`torch.compile`的默认设置，但遇到图断裂后恢复编译的语义更为复杂。有关`fullgraph=False`语义的详细信息，请参考各子章节。

使用`torch.compile(fullgraph=False)`时，建议采用以下策略：

1. [确定应用`torch.compile`的理想位置](where_to_apply_compile.md)。通常应选择不会产生过多图断裂的最高层函数。执行大量预处理或I/O操作的函数容易产生许多图断裂，也无法从`torch.compile`中显著获益。a. 可以先编译单个函数或模块来隔离问题，然后再编译整个模型。
2. [对编译区域内会产生大量图断裂且无法从编译中获益的函数应用`torch.compiler.disable`](compiler_disable.md)。在这种情况下，一个图断裂优于潜在的数十或数百个图断裂。
3. [使用`TORCH_LOGS="graph_breaks"`或tlparse调查其余图断裂。](observability.md)采用与`fullgraph=True`编程模型相同的方法绕过这些图断裂。并非所有图断裂都需要消除，有些图断裂对性能的影响更大。一般应重点关注模型计算期间发生的图断裂。a. 调试图断裂时，建议使用`torch.compile(backend='eager')`，以加快调试迭代速度。

- [`torch.compile`的应用位置](where_to_apply_compile.md)
- [禁用编译和抑制错误](compiler_disable.md)
- [切换`error_on_graph_break`](error_on_graph_break.md)
- [嵌套图断裂](nested_graph_breaks.md)
- [跳过的函数](skipped_functions.md)
