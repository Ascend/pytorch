# 概述

Ascend Extension for PyTorch插件是基于昇腾的深度学习适配框架，使昇腾NPU可以支持PyTorch框架，为PyTorch框架的使用者提供昇腾AI处理器的超强算力。

该插件最大限度地继承了PyTorch框架的动态图特性、原生开发方式以及体系结构，使得开发者在使用Ascend Extension for PyTorch时，几乎无需改变原有的开发习惯和代码风格。开发者可以继续使用熟悉的PyTorch接口和函数，只需在指定设备为昇腾NPU的情况下，即可将模型无缝迁移到昇腾平台上进行训练，大大降低了开发成本和迁移难度。

随着深度学习领域蓬勃发展，数据规模不断膨胀，模型复杂度的持续攀升，对计算平台的挑战日益增长。面对这些挑战，Ascend Extension for PyTorch从内存资源优化、通信性能优化、计算性能优化、辅助报错定位等方面精心打造了一系列独特的特性。为用户提供了一个高效、便捷的开发工具，具体特性请参见[表1](#特性列表)。

**表 1**  特性列表 <a id="特性列表"></a>

|特性类别|特性名称|特性说明|
|--|--|--|
|内存资源优化|[虚拟内存](virtual_memory.md)|动态调整内存块的大小，减少内存碎片的产生。|
|内存资源优化|[内存快照](memory_snapshot.md)|支持训练过程中内存溢出时生成设备内存快照。|
|内存资源优化|[自定义内存分配器](custom_memory_allocator.md)|从so文件加载自定义NPU内存分配器。|
|内存资源优化|[多流内存复用](multistream_memory_reuse.md)|多流情况下，优化内存使用情况，提高内存复用率。|
|内存资源优化|[内存共享（IPC）](memory_sharing_ipc.md)|支持跨进程共享内存，有效减少内存消耗。|
|通信性能优化|[torch_npu_run](torch_npu_run.md)|torch_npu_run是torchrun在大集群场景的改进版，提升集群建链性能。|
|通信性能优化|[ranktable建链](ranktable_link_setup.md)|支持以ranktable文件配置方式建立通信域。|
|计算性能优化|[自动绑核](automatic_core_binding.md)|通过设置粗/细粒度绑核，优化Ascend Extension for PyTorch下发性能。|
|计算性能优化|[Stream级TaskQueue并行下发](stream_taskqueue_parallel_delivery.md)|每个Stream会初始化独立的TaskQueue和对应的Dequeue线程，实现真正的二级流水并行下发机制。|
|辅助报错定位|[特征值检测](feature_value_detection.md)|基于通信流做静默数据错误的特征值检测，识别精度问题。|
|辅助报错定位|[WatchDog](watchdog.md)|在不影响大模型训练性能和精度的前提下，能快速稳定发现错误。|
|参数配置|[通过pg_options配置HCCL通信域参数](setting_HCCL_communicator_parameter.md)|可以针对不同的通信域配置不同的HCCL配置。|
|PyTorch图模式|[PyTorch编译模式（torch.compile）](pytorch_compilation_mode.md)|Ascend Extension for PyTorch在2.6.0以上版本已支持torch.compile()，通过“动态图捕获+静态图优化+高效代码生成”的方式显著加速模型训练和推理任务。|
|算子适配|[自定义算子适配开发](custom_operator_adaptation.md)|基于OpPlugin插件或C++ extensions的方式编写并调用自定义算子。|


