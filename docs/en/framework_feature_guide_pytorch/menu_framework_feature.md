# Framework Feature Guide

-   [Overview](overview.md)
-   [Memory Resource Optimization](./memory_resource_optimization.md)
    -   [Virtual Memory](./virtual_memory.md)
    -   [Memory Snapshot](./memory_snapshot.md)
    -   [Custom Memory Allocator](./custom_memory_allocator.md)
    -   [Multi-Stream Memory Reuse](./multistream_memory_reuse.md)
    -   [Memory Sharing (IPC)](./memory_sharing_ipc.md)

-   [Communication Performance Optimization](./communication_performance_optimization.md)
    -   [torch\_npu\_run](torch_npu_run.md)
    -   [Ranktable link setup](./ranktable_link_setup.md)

-   [Computing Performance Optimization](./computing_performance_optimization.md)
    -   [Automatic Core Binding](./automatic_core_binding.md)
    -   [Stream-Level TaskQueue Parallel Delivery](./stream_taskqueue_parallel_delivery.md)

-   [Assisted Error Location](./assisted_error_locating.md)
    -   [Feature Value Detection](./feature_value_detection.md)
    -   [WatchDog](./watchdog.md)

-   [Parameter Configuration](./parameter_setting.md)
    -   [Configuring HCCL Communication Domain Parameters Through pg_options](./setting_HCCL_communicator_parameter.md)

-   [Custom Operator Adaptation](./custom_operator_adaptation.md)
    -   [OpPlugin Operator Adaptation](./opplugin_operator_adaptation.md)
        -   [Overview](./adaptation_overview_opplugin.md)
        -   [Operator Adaptation Process](./adaptation_flow_opplugin.md)
        -   [OpPlugin Operator Adaptation](./opplugin_operator_adaptation.md)
            -   [Preparation Before Adaptation](./adaptation_preparation_opplugin.md)
            -   [Adaptation Development](./adaptation_development_opplugin.md)
            -   [Compilation Verification](./adaptation_compile_opplugin.md)
        -   [Sample Call](./sample_call_opplugin.md)
        -   [Common References](./reference.md)
  
    -   [C++ Extension Operator Adaptation](./c_extensions_operator_adaptation.md)
        -   [Adaptation Description](https://gitcode.com/Ascend/op-plugin/blob/master/examples/README.md)
        -   [Adaptation Development and Usage (Basic Example)](https://gitcode.com/Ascend/op-plugin/blob/master/examples/cpp_extension_base/README.md)
        -   [Adaptation Development and Usage (Complete Module Example)](https://gitcode.com/Ascend/op-plugin/blob/master/examples/cpp_extension_full/module/README.md)
        -   [Adaptation Development and Usage (Complete TORCH_LIBRARY_IMPL Example)](https://gitcode.com/Ascend/op-plugin/blob/master/examples/cpp_extension_full/torch_lib_impl/README.md)
        -   [Adaptation Development and Usage (Structured)](https://gitcode.com/Ascend/op-plugin/blob/master/examples/cpp_extension_structured/README.md)
        -   [Adaptation Development and Usage (AscendC-based)](https://gitcode.com/Ascend/op-plugin/blob/master/examples/cpp_extension_asc/README.md)
        -   [Adaptation Development and Usage (pybind-based)](https://gitcode.com/Ascend/op-plugin/blob/master/examples/cpp_extension_pybind/README.md)
