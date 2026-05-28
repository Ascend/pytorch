# 概述

Ascend Extension for PyTorch插件是基于昇腾的深度学习适配框架，使昇腾NPU可以支持PyTorch框架，为PyTorch框架的使用者提供昇腾AI处理器的超强算力。

该插件最大限度地继承了PyTorch框架的动态图特性、原生开发方式以及体系结构，使得开发者在使用Ascend Extension for PyTorch时，几乎无需改变原有的开发习惯和代码风格。开发者可以继续使用熟悉的PyTorch接口和函数，只需在指定设备为昇腾NPU的情况下，即可将模型无缝迁移到昇腾平台上进行训练，大大降低了开发成本和迁移难度。

随着深度学习领域蓬勃发展，数据规模不断膨胀，模型复杂度的持续攀升，对计算平台的挑战日益增长。面对这些挑战，Ascend Extension for PyTorch从内存资源优化、通信性能优化、计算性能优化、辅助报错定位等方面精心打造了一系列独特的特性。为用户提供了一个高效、便捷的开发工具，具体特性请参见[表1](#特性列表)。

**表 1**  特性列表 <a id="特性列表"></a>
<table style="undefined;table-layout: fixed; width: 1508px"><colgroup>
<col style="width: 293px">
<col style="width: 252px">
<col style="width: 963px">
</colgroup>
<thead>
  <tr>
    <th>特性类别<br></th>
    <th>特性名称<br></th>
    <th>特性说明<br></th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="5">内存资源优化</td>
    <td><a href="./virtual_memory.md">虚拟内存</a></td>
    <td>动态调整内存块的大小，减少内存碎片的产生。</td>
  </tr>
  <tr>
    <td><a href="./memory_snapshot.md">内存快照</a></td>
    <td>支持训练过程中内存溢出时生成设备内存快照。</td>
  </tr>
  <tr>
    <td><a href="./custom_memory_allocator.md">自定义内存分配器</a></td>
    <td>从.so文件加载自定义NPU内存分配器。</td>
  </tr>
  <tr>
    <td><a href="./multistream_memory_reuse.md">多流内存复用</a></td>
    <td>多流情况下，优化内存使用情况，提高内存复用率。</td>
  </tr>
  <tr>
    <td><a href="./memory_sharing_ipc.md">内存共享（IPC）</a></td>
    <td>支持跨进程共享内存，有效减少内存消耗。</td>
  </tr>
  <tr>
    <td rowspan="2">通信性能优化</td>
    <td><a href="./torch_npu_run.md">torch_npu_run</a></td>
    <td>torch_npu_run是torchrun在大集群场景的改进版，提升集群建链性能。</td>
  </tr>
  <tr>
    <td><a href="./ranktable_link_setup.md">ranktable建链</a></td>
    <td>支持以ranktable文件配置方式建立通信域。</td>
  </tr>
  <tr>
    <td rowspan="2">计算性能优化</td>
    <td><a href="./automatic_core_binding.md">自动绑核</a></td>
    <td>通过设置粗/细粒度绑核，优化Ascend Extension for PyTorch下发性能。</td>
  </tr>
  <tr>
    <td><a href="./stream_taskqueue_parallel_delivery.md">Stream级TaskQueue并行下发</a></td>
    <td>每个Stream会初始化独立的TaskQueue和对应的Dequeue线程，实现真正的二级流水并行下发机制。</td>
  </tr>
  <tr>
    <td rowspan="2">辅助报错定位</td>
    <td><a href="./feature_value_detection.md">特征值检测</a></td>
    <td>基于通信流做静默数据错误的特征值检测，识别精度问题。</td>
  </tr>
  <tr>
    <td><a href="./watchdog.md">WatchDog</a></td>
    <td>在不影响大模型训练性能和精度的前提下，能快速稳定发现错误。</td>
  </tr>
  <tr>
    <td>参数配置</td>
    <td><a href="./setting_HCCL_communicator_parameter.md" target="_blank" rel="noopener noreferrer">通过pg_options配置HCCL通信域参数</a></td>
    <td>可以针对不同的通信域配置不同的HCCL配置。</td>
  </tr>
  <tr>
    <td>torch_npu.npu.NPUGraph</td>
    <td><a href="./pytorch_npugraph_desc.md">torch_npu.npu.NPUGraph</a></td>
    <td>NPUGraph是一种在Eager Mode（单算子执行模式）下使用的静态图捕获技术，将一系列NPU内核定义并封装为一个单元（即操作图），通过单一CPU操作启动多个NPU操作，从而减少启动开销。</td>
  </tr>
  <tr>
    <td>torch.compile</td>
    <td><a href="./pytorch_graph_mode.md">torch.compile</a></td>
    <td>torch.compile是PyTorch 2.0推出的核心编译接口，通过“动态图捕获+静态图优化+高效代码生成”的方式显著加速模型训练和推理任务。</td>
  </tr>
  <tr>
    <td>算子适配</td>
    <td><a href="./custom_operator_adaptation.md">自定义算子适配开发</a></td>
    <td>基于OpPlugin插件或C++ extensions的方式编写并调用自定义算子。</td>
  </tr>
</tbody></table>
