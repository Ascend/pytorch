# Overview

`torch.compile()` is the core compilation interface introduced in PyTorch 2.0. It significantly accelerates model training and inference tasks through "dynamic graph capture + static graph optimization + efficient code generation". TorchNPU has supported `torch.compile()` since version 7.3.0. With just one line of code, it automatically implements frontend graph capture and backend optimization, making it suitable for fully automatic compilation scenarios.

torch.compile includes the following core components:

| Component | Function |
|-------------------|-----------|
| Dynamo frontend | Dynamo can JIT (just-in-time) compile the user's eager (dynamic graph) code into FX Graph (PyTorch's intermediate representation). |
| AOT Autograd | Captures the backward propagation graph in advance, allowing both forward and backward propagation to be optimized by the backend.|
| Compilation backend | Optimizes the FX Graph and generates the final executable code.|

For general usage and core concepts, see the [Reference Documentation](#reference-documentation) section in this document.

## Interface Description

### Interface Prototype

```python
torch.compile(model, *, fullgraph=False, dynamic=None, backend="inductor",
              mode=None, options=None, disable=False)
```

### Parameter Description

| Parameter | Data Type | Default Value | Description |
|------|------|--------|------|
| model | nn.Module | Required | The model to be compiled |
| fullgraph | bool | False | Whether to force whole-graph compilation |
| dynamic | bool | None | Whether to enable dynamic shape compilation |
| backend | str/Callable | `"inductor"` | Compilation backend: `inductor`, `npugraphs`, `npugraph_ex`, `aot_eager`, `TorchAir-GE backend (Callable)` |
| mode | str | None | Compilation mode: `None` or `"reduce-overhead"` (supported only by the `inductor` backend)|
| options | dict | None | Compilation options|
| disable | bool | False | Disables torch.compile |

For more parameter details, see [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

**Compilation backend description**

| Backend | How to Enable | Core Mechanism | Applicable Scenarios |
|------|----------|---------|---------|
| Inductor (default) | `backend="inductor"` | Operator fusion + code generation (Triton/DVM/Ascend C) | Most scenarios; the first choice when unsure |
| NPUGraphs | `backend="npugraphs"` | ACLGraph graph sinking: capture once and replay multiple times, eliminating kernel launch overhead | Frequent kernel calls and CPU scheduling intensive |
| NPUGraph_EX | `backend="npugraph_ex"` | ACLGraph graph sinking + FX graph optimization + compilation cache reuse | Large model inference deployment |
| AOT_Eager | `backend="aot_eager"` | Performs no optimization; only verifies graph capture correctness | Debugging and baseline performance comparison |
| TorchAir-GE | `backend=torchair.get_npu_backend(...)` | Converts the PyTorch FX graph into a computation graph and implements computation graph compilation and execution through the GE graph engine | Large model inference deployment |

## Reference Documentation

The following table organizes the sections about [torch.compile](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html) in the official PyTorch documentation by feature and usage scenario, covering general usage, detailed working mechanisms, and common problem troubleshooting guides for quick retrieval.

| Reference Documentation | Introduction |
| --- | --- |
| [Quick Start](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_get_started.html) | Introduces the basic usage of torch.compile. Using simple functions and pretrained model examples, it shows how to view the compiled code, helping you gain an in-depth understanding of operator fusion. |
| [Dynamo Overview](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_dynamo_overview.html) | Analyzes the process by which Dynamo extracts FX graphs from Python bytecode and invokes backend compilation, using examples to explain the Guard verification mechanism and how to view compilation artifacts. |
| [torch.compile Programming Model](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.html) | Provides an overview of compilation behavior and fine-grained control strategies, serving as an entry point to core documents such as graph break handling, non-strict tracing, recompilation optimization, and debugging methods. |
| [Dynamo Core Concepts](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html) | Explains the core mechanisms around Dynamo tracing, graph breaks, Guards, and recompilation, using examples to illustrate the role of dynamic shapes in reducing recompilation. |
| [Handling Graph Breaks](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html) | Introduces the compilation, execution, and trace recovery process on graph breaks, analyzes its impact on performance, and summarizes handling strategies across different compilation modes. |
| [Using fullgraph=True to Identify and Eliminate Graph Breaks](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html) | Explains how to use fullgraph=True to locate graph breaks, and the specific methods to eliminate them through code refactoring, enabling non-strict tracing, custom operators, or adjusting compilation regions. |
| [Common Graph Breaks](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html) | Analyzes graph breaks caused by code errors, data-dependent operations, and log printing through examples, and provides corresponding troubleshooting steps and workarounds. |
| [Using nonstrict_trace](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.dynamo_nonstrict_trace.html) | Introduces how to use nonstrict_trace to trace functions within compilation regions, and explains the input and output type constraints and the constant handling rules for non-input values. |
| [Custom Operators](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.custom_ops.html) | Introduces wrapping hard-to-trace functions with custom operators so that the compiler preserves the call without tracing the internal implementation, and provides an entry point for usage guidance on Python and C++ custom operators. |
| [Using fullgraph=False](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.fullgraph_false.html) | Introduces compilation strategies in fullgraph=False mode, including selecting compilation entry points, disabling functions unsuitable for compilation, and how to use logs to identify key graph breaks affecting performance. |
| [Where to Apply torch.compile](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.where_to_apply_compile.html) | Provides best practices for applying compilation to training steps, top-level modules, and submodules, explaining how to use `model.compile()` and recommendations for handling distributed wrapper modules. |
| [Disabling Compilation and Suppressing Errors](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.compiler_disable.html) | Introduces how to disable local compilation with torch.compiler.disable, and explains the control strategy for recursive calls as well as the behavior and limitations of suppressing compilation errors. |
| [Switching error_on_graph_break](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.error_on_graph_break.html) | Explains how to set graph break error behavior per code region in fullgraph=False mode, covering nested setting rules and its relationship with fullgraph=True. |
| [Nested Graph Breaks](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.nested_graph_breaks.html) | Uses nested function examples to walk through the graph break handling and trace recovery process, analyzing the causes of repeated tracing and the additional overhead of nested calls. |
| [Skipped Functions](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.skipped_functions.html) | Explains the reasons why functions in loops, context managers, and try statement blocks are skipped during compilation, and provides methods to fix graph breaks or isolate problematic code. |
| [Non-Strict Tracing Programming Model](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.non_strict_tracing_model.html) | Introduces the working mechanism of non-strict tracing, explains pure function constraints, data pointer operation restrictions, and constant specialization behavior, and analyzes with examples why tracing results may differ from the original function. |
| [Handling Recompilation](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.recompilation.html) | Introduces methods to locate the causes of recompilation, and optimization strategies such as dynamic shape optimization, scalar tensorization, module integer attribute configuration, and cache limit adjustment to help reduce recompilation overhead. |
| [Reducing Guard Overhead](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.reducing_guard_overhead.html) | Analyzes the bytecode overhead of Guard verification before computation graph execution, provides optimization methods, and explains the applicable conditions and correctness risks of Guard filtering and skipping verification. |
| [tlparse / TORCH_TRACE](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.observability.html) | Explains TORCH_TRACE log collection, tlparse compilation report parsing, and TORCH_LOGS log configuration methods, helping to locate issues such as graph breaks, Guard verification, and recompilation. |
| [Reporting Issues](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.reporting_issues.html) | Organizes the process for locating and reporting compilation issues, covering backend comparison, version bisection troubleshooting, and independent reproduction example construction, helping to prepare issue reports that are easy to analyze. |
| [PyTorch 2.0 NNModule Support](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_nn_module.html) | Explains torch.compile's special handling mechanism for nn.Module, focusing on the support status, usage restrictions, and change detection configuration of Hooks related to forward, backward, and state_dict. |
| [Autograd Semantic Differences of torch.compile](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_backward.html) | Introduces the Autograd semantic differences between compiled mode and dynamic graph mode, focusing on the autocast assumptions in backward propagation and how to configure backward_pass_autocast. |
