# 框架特性指南

-   [概述](overview.md)
-   [内存资源优化](./memory_resource_optimization.md)
    -   [虚拟内存](./virtual_memory.md)
    -   [内存快照](./memory_snapshot.md)
    -   [自定义内存分配器](./custom_memory_allocator.md)
    -   [多流内存复用](./multistream_memory_reuse.md)
    -   [内存共享（IPC）](./memory_sharing_ipc.md)


-   [通信性能优化](./communication_performance_optimization.md)
    -   [torch\_npu\_run](torch_npu_run.md)
    -   [ranktable建链](./ranktable_link_setup.md)

-   [计算性能优化](./computing_performance_optimization.md)
    -   [自动绑核](./automatic_core_binding.md)
    -   [Stream级TaskQueue并行下发](./stream_taskqueue_parallel_delivery.md)

-   [辅助报错定位](./assisted_error_locating.md)
    -   [特征值检测](./feature_value_detection.md)
    -   [WatchDog](./watchdog.md)

-   [参数配置](./parameter_setting.md)
    -   [通过pg\_options配置HCCL通信域参数](./setting_HCCL_communicator_parameter.md)

-   [PyTorch图模式](./pytorch_graph_mode.md)
    -   [PyTorch编译模式（torch.compile）](./pytorch_compilation_mode.md)

-   [自定义算子适配开发](./custom_operator_adaptation.md)
    -   [基于OpPlugin算子适配开发](./opplugin_operator_adaptation.md)
        -   [概述](./adaptation_overview_opplugin.md)
        -   [算子适配流程](./adaptation_flow_opplugin.md)
        -   [opplugin算子适配]()
            -   [适配前准备](./adaptation_preparation_opplugin.md)
            -   [适配开发](./adaptation_development_opplugin.md)
            -   [编译验证](./adaptation_compile_opplugin.md)                                    
        -   [调用样例](./sample_call_opplugin.md)
        -   [常见参考](./reference.md)    

    -   [基于C++ extensions算子适配开发](./c_extensions_operator_adaptation.md)
        -   [适配说明](./adaptation_description_extension.md)
        -   [单算子API调用适配](./single_operator_adaptation.md)
            -   [适配开发](./adaptation_description_single.md)
            -   [调用样例](./sample_call_single.md)

        -   [kernel直调算子适配](./kernel_launch_operator_adaptation.md)
            -   [适配开发](./adaptation_description_kernel.md)
            -   [调用样例](./sample_call_kernel.md)

