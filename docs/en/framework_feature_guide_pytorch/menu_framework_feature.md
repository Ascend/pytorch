
# Framework Features

-   [Overview](overview.md)
-   [Memory Resource Optimization](./memory_resource_optimization.md)
    -   [Virtual Memory](./virtual_memory.md)
    -   [Memory Snapshot](./memory_snapshot.md)
    -   [Custom Memory Allocator](./custom_memory_allocator.md)
    -   [Multi-Stream Memory Reuse](./multistream_memory_reuse.md)
    -   [Memory Sharing (IPC)](./memory_sharing_ipc.md)


-   [Communication Performance Optimization](./communication_performance_optimization.md)
    -   [torch_npu_run](torch_npu_run.md)
    -   [Ranktable link setup](./ranktable_link_setup.md)

-   [Computing Performance Optimization](./computing_performance_optimization.md)
    -   [Automatic Core Binding](./automatic_core_binding.md)
    -   [Stream-Level TaskQueue Parallel Delivery](./stream_taskqueue_parallel_delivery.md)
    -   [Compilation Optimization](comp_opt.md)
        -   [Compilation Optimization Technology Introduction](comp_opt_intro.md)
        -   [Installing the Bisheng Compiler](install_bisheng_comp.md)
        -   [Compilation Optimization (Python)](comp_opt_py.md)
        -   [Compilation Optimization (PyTorch)](pytorch_comp_opt.md)
        -   [Compilation Optimization (TorchNPU)](torch_npu_comp_opt.md)
        -   [Compilation Optimization FAQ](comp_opt_faq.md)

-   [Assisted Error Location](./assisted_error_locating.md)
    -   [Feature Value Detection](./feature_value_detection.md)
    -   [WatchDog](./watchdog.md)

-   [Parameter Configuration](./parameter_setting.md)
    -   [Configuring HCCL Communication Domain Parameters Through pg_options](./setting_HCCL_communicator_parameter.md)

-   [torch_npu.npu.NPUGraph](./pytorch_npugraph_desc.md)

-   [Custom Operator Adaptation Development](./custom_operator_adaptation.md)
    -   [Operator Adaptation Development Based on OpPlugin](./opplugin_operator_adaptation.md)
        -   [Overview](./adaptation_overview_opplugin.md)
        -   [Operator Adaptation Process](./adaptation_flow_opplugin.md)
        -   [OpPlugin Operator Adaptation]()
            -   [Preparation Before Adaptation](./adaptation_preparation_opplugin.md)
            -   [Adaptation Development](./adaptation_development_opplugin.md)
            -   [Compilation Verification](./adaptation_compile_opplugin.md)
        -   [Sample Call](./sample_call_opplugin.md)
        -   [Common References](./reference.md)

    -   [Operator Adaptation Development Based on C++ Extensions](./c_extensions_operator_adaptation.md)
        -   [Adaptation Description](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/examples/README_en.md)
        -   [Adaptation Development and Usage (Basic Example)](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/examples/cpp_extension_base/README_en.md)
        -   [Adaptation Development and Usage (Complete Example - Module)](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/examples/cpp_extension_full/module/README_en.md)
        -   [Adaptation Development and Usage (Complete Example - TORCH_LIBRARY_IMPL)](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/examples/cpp_extension_full/torch_lib_impl/README_en.md)
        -   [Adaptation Development and Usage (Structured)](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/examples/cpp_extension_structured/README_en.md)
        -   [Adaptation Development and Usage (AscendC)](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/examples/cpp_extension_asc/README_en.md)
        -   [Adaptation Development and Usage (pybind)](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/examples/cpp_extension_pybind/README_en.md)
