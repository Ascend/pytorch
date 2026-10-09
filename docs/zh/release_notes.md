# 版本说明

## 关键特性

- 支持alltoallvc通信算子，优化alltoall通信性能。
- 支持通信域以分级建链方式初始化，性能优化。
- 支持Inductor基于AscendC IR的自动算子融合。
- 继承DeviceAllocator，支持在Eager模式下启用accelerator memory和graph相关接口。

## 版本配套说明

### 产品版本信息

<table><tbody><tr id="row135479428341"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.1.1"><p id="p125478428345">产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.1.1 "><p id="p3547142103415"><span id="ph4778145519911">TorchNPU</span></p>
</td>
</tr>
<tr id="row11547114203412"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.2.1"><p id="p17547142103418">产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.2.1 "><p id="p2547184216342"><span id="ph1414342615376">26.2.0</span></p>
</td>
</tr>
<tr id="row854711422349"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.3.1"><p id="p354754216341">版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.3.1 "><p id="p2547114214349">正式版本</p>
</td>
</tr>
<tr id="row754461214611"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.4.1"><p id="p155445122062">发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.4.1 "><p id="p135443128613">2026年10月</p>
</td>
</tr>
<tr id="row954744243418"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.5.1"><p id="p15471742193419">维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.5.1 "><p id="p1154734212344">参考<a href="https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/README.zh.md#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5">分支维护策略</a></p>
</td>
</tr>
</tbody>
</table>

### 相关产品版本配套说明

固件和驱动的版本配套表与所有的昇腾硬件及CANN版本相关，具体选择请参考[CANN版本说明](https://gitcode.com/cann/release-management/blob/master/9.2.0/release-notes.md)。

为扩展TorchNPU能力，昇腾提供的自研插件，其版本要求说明请参考[配套软件库](./user_guide/libraries.md)。

TorchNPU代码分支名称采用 **\{PyTorch版本\}-\{TorchNPU版本\}** 的命名规则，前者为TorchNPU匹配的PyTorch版本，详细匹配如下表：

|TorchNPU代码分支名称|PyTorch版本|TorchNPU版本|TorchNPU安装包版本|CANN版本|Python版本|Triton Ascend版本|
|--|--|--|--|--|--|--|
|v2.7.1-26.2.0|2.7.1|26.2.0|2.7.1.post11|9.2.0|Python3.9.*x*、Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|v3.2.2|
|v2.9.0-26.2.0|2.9.0|26.2.0|2.9.0.post9|9.2.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*、Python3.14.*x*|无配套版本|
|v2.10.0-26.2.0|2.10.0|26.2.0|2.10.0.post7|9.2.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*、Python3.14.*x*|v3.6.0|
|v2.11.0-26.2.0|2.11.0|26.2.0|2.11.0.post3|9.2.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*、Python3.14.*x*|无配套版本|
|v2.12.0-26.2.0|2.12.0|26.2.0|2.12.0.post3|9.2.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*、Python3.14.*x*|无配套版本|
|v2.13.0-26.2.0|2.13.0|26.2.0|2.13.0|9.2.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*、Python3.14.*x*|v3.6.0|
|v2.14.0-26.2.0|2.14.0|26.2.0|2.14.0|9.2.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*、Python3.14.*x*|无配套版本|

## 版本兼容性说明

> [!NOTE]
>
> 表格中“Y”表示兼容。

<p style="display:none">
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg .tg-rhr9{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style>
</p>
<table class="tg"><thead>
  <tr>
    <th class="tg-rhr9" rowspan="2">TorchNPU</th>
    <th class="tg-amwm" colspan="4">CANN版本</th>
  </tr>
  <tr>
    <th class="tg-c3ow">8.5.X</th>
    <th class="tg-c3ow">9.0.X</th>
    <th class="tg-c3ow">9.1.X</th>
    <th class="tg-c3ow">9.2.X</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">7.3.X</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
  </tr>
  <tr>
    <td class="tg-baqh">26.0.X</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
  </tr>
    <tr>
    <td class="tg-baqh">26.1.X</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
  </tr>
  <tr>
    <td class="tg-baqh">26.2.X</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
  </tr>
</tbody>
</table>

## 更新说明

### 新增特性说明

<table>
  <thead align="left">
    <tr>
      <th class="cellrowborder" valign="top" width="18.801880188018803%" id="mcps1.1.4.1.1">组件</th>
      <th class="cellrowborder" valign="top" width="32.603260326032604%" id="mcps1.1.4.1.2">特性</th>
      <th class="cellrowborder" valign="top" width="48.5948594859486%" id="mcps1.1.4.1.3">特性说明</th>
      <th class="cellrowborder" valign="top" width="32.603260326032604%" id="mcps1.1.4.1.2">适配说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="cellrowborder" rowspan="19" valign="top" width="18.801880188018803%" headers="mcps1.1.4.1.1">TorchNPU</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">DTensor补全。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">math ops分片策略全覆盖；新增DeviceMesh上的linspace工厂函数。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">新增适配PyTorch 2.14.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">c10d架构标准化。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">后端发现机制重构为Python entrypoints动态注册。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">新增适配PyTorch 2.14.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">通信容错。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">NCCL2新增reconfigure能力，支持重建通信子做故障恢复。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">新增适配PyTorch 2.14.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">编译配置简化。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">支持全局设置默认compile后端。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">新增适配PyTorch 2.14.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">算子覆盖扩展。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">uniform进入编译器分解路径；新增shallow_copy_data_支持跨设备tensor.data=赋值（Dynamo到Inductor全栈）。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">新增适配PyTorch 2.14.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">FSDP2分片灵活性增强。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">支持逐参数在不同DeviceMesh上独立分片； fully_shard可通过 DataParallelMeshDims在SPMD网格上显式声明数据并行维度。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">新增适配PyTorch 2.13.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">FSDP2 chunked loss。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">norm/head部分前向，降低大模型训练峰值显存。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">新增适配PyTorch 2.13.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">量化支持扩展。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">AOTInductor C shim层支持MXFP4。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">新增适配PyTorch 2.13.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">重编译控制。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">新增recompile_limit，调用级限制单函数重编译次数。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">新增适配PyTorch 2.13.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">算子覆盖扩展。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">inplace自定义算子支持compile；scan输出维度movedim；BatchLinearLHSFusion识别更多linear调用。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">新增适配PyTorch 2.13.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Mempool社区特性跟进。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">架构跟随社区主线，修复显存泄露问题，支持单独的ID控制机制，补齐社区新增特性。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">新增适配PyTorch 2.13.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Eager模式API支持torch.accelerator。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">继承DeviceAllocator，支持在Eager模式下启用accelerator memory和graph相关接口。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">新增适配PyTorch 2.13.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">torch.bitwise_right_shift和torch.bitwise_left_shift支持NPU。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">torch.bitwise_right_shift和torch.bitwise_left_shift支持在NPU上计算，避免fallback回退至CPU。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">适配PyTorch 2.12.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">支持Inductor AscendC后端。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">支持Inductor基于AscendC IR的自动算子融合。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">适配PyTorch 2.11.0及以上版本</td>
    </tr>
      <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">内存优化配置标准化。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">重构内存配置项，复用社区通用配置并通过注册机制实现；支持通过PYTORCH_ALLOC_CONF环境变量进行动态配置。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">适配PyTorch 2.10.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">支持alltoallvc通信算子。</td>
      <td class="cellrowborder" valign="top" width="48.5948594859486%" headers="mcps1.1.4.1.3">支持alltoallvc通信算子，优化alltoall通信性能。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">&#8226; 适配PyTorch 2.7.1版本<br>&#8226; 适配PyTorch 2.9.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">通信域支持分级建链初始化。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">支持通信域以分级建链方式初始化，优化性能。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">&#8226; 适配PyTorch 2.7.1版本<br>&#8226; 适配PyTorch 2.9.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">通信算子支持对称内存。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">支持内存窗口注册，关键通信算子支持对称内存，提升数据传输效率。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">&#8226; 适配PyTorch 2.7.1版本<br>&#8226; 适配PyTorch 2.9.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">支持Batch一致性配置。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">torch_npu.npu.set_deterministic_level(level)接口新增枚举值3，用于指定Batch级别的一致性； 通过get接口返回当前的level情况。确定性计算与Batch一致性均支持动态配置。</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">&#8226; 适配PyTorch 2.7.1<br>&#8226; 适配PyTorch 2.9.0及以上版本</td>
    </tr>
  </tbody>
</table>

### 关键特性变更

本版本继承了本产品26.1.X版本的所有特性。

### 接口变更说明

本章节的接口变更说明包括新增、修改、废弃和删除。接口变更只体现代码层面的修改，不包含文档本身在语言、格式、链接等方面的优化改进。

- 新增：表示此次版本新增的接口。
- 修改：表示本接口相比于上个版本有修改。
- 废弃：表示该接口自作出废弃声明的版本起停止演进，且在声明一年后可能被移除。
- 删除：表示该接口在此次版本被移除。

**表 1** TorchNPU接口变更汇总

<table>
  <thead align="left">
    <tr>
      <th class="cellrowborder" valign="top" width="11.53%" id="mcps1.2.6.1.1">变更版本</th>
      <th class="cellrowborder" valign="top" width="37.68%" id="mcps1.2.6.1.2">类名/API原型</th>
      <th class="cellrowborder" valign="top" width="15.22%" id="mcps1.2.6.1.3">类/API类别</th>
      <th class="cellrowborder" valign="top" width="15.32%" id="mcps1.2.6.1.4">变更类别</th>
      <th class="cellrowborder" valign="top" width="20.25%" id="mcps1.2.6.1.5">变更说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="cellrowborder" rowspan="216" valign="top" width="11.53%" headers="mcps1.2.6.1.1">v2.7.1</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.batch_norm_reduce</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.distributed.all_to_all_vc</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_quant_gmm_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm_cast</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm_dynamic_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_all_gather_quant_mm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_all_to_all_quant_matmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_anti_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.2.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_apply_adam_w</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dual_level_quant_matmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_block_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_mx_quant_with_dual_axis</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_cross_entropy_loss_with_max_sum</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_linear_online_max_sum</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_matmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fusion_attention_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_gelu_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_dynamic_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_add</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_add_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_swiglu_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_kronecker_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_kv_rmsnorm_rope_cache_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_token_permute</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.2.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_token_permute_with_routing_map</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">&#8226; 在<term>Atlas A2系列产品</term>/<term>Atlas A3系列产品</term>上依赖CANN 8.5.0及以上版本<br>&#8226; 在<term>Ascend 950PR&950DT系列产品</term>上依赖CANN 9.2.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_token_unpermute</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.2.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_token_unpermute_with_routing_map</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">&#8226; 在<term>Atlas A2系列产品</term>/<term>Atlas A3系列产品</term>上依赖CANN 8.5.0及以上版本<br>&#8226; 在<term>Ascend 950PR&950DT系列产品</term>上依赖CANN 9.2.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_gmm_alltoallv</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_matmul_all_to_all</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_mm_reduce_scatter</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>   
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rms_norm_dynamic_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0及以上版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rms_norm_quant_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0及以上版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_scatter_pa_cache</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_swiglu_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_weight_quant_preprocess</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.2.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.get_task_queue_enable</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_task_queue_enable</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.NPUPluggableAllocator</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.change_current_allocator</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.distributed.reduce_scatter_tensor_uneven</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.empty_with_swapped_memory</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.erase_stream</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.get_device_limit</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.get_stream_limit</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.matmul_checksum</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.ExternalEvent</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.SyncLaunchStream</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.aclnn.allow_hf32</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.are_compatible_impl_enabled</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.conv.allow_hf32</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.empty_virt_addr_cache</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.graph_task_group_begin</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.graph_task_group_end</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.graph_task_update_begin</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.graph_task_update_end</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.host_empty_cache</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.matmul.allow_hf32</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.matmul.cube_math_type</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.mstx.mark</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.mstx.mstx_range</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.mstx.range_end</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.mstx.range_start</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.mstx</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_deterministic_level</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_op_timeout_ms</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.use_compatible_impl</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm_dynamic_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_all_gather_base_mm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_alltoallv_gmm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_anti_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_block_sparse_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_chunk_gated_delta_rule</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.2.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_clipped_swiglu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_convert_weight_to_int4pack</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dense_lightning_indexer_grad_kl_loss</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dense_lightning_indexer_softmax_lse</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dequant_swiglu_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_block_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_quant_asymmetric</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fast_gelu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_infer_attention_score</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_infer_attention_score_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_gelu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_gmm_alltoallv</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_group_norm_silu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_group_norm_swish</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_finalize_routing</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_interleave_rope</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_kv_quant_sparse_flash_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.2.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_kv_rmsnorm_rope_cache</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mm_all_reduce_base</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mm_reduce_scatter_base</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_compute_expert_tokens</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_distribute_combine</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_distribute_combine_add_rms_norm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_distribute_combine_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_distribute_dispatch</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_distribute_dispatch_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_finalize_routing</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 8.5.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_gating_top_k</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 8.5.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_gating_top_k_softmax</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_init_routing</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 8.5.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_init_routing_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 8.5.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_update_expert</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mrope</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.NpuGraphOpHandler</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_matmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_scatter</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_scatter_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quantize</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_recurrent_gated_delta_rule</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rotary_mul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rotate_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.2.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_scatter_nd_update</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_scatter_nd_update_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_scatter_pa_kv_cache</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_sim_exponential_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_sparse_lightning_indexer_grad_kl_loss</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_trans_quant_param</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_transpose_batchmatmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_weight_quant_batchmatmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedAdadelta</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedAdam</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedAdamP</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedAdamW</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedBertAdam</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedLamb</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedOptimizerBase</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedRMSprop</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedRMSpropTF</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedSGD</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.AiCMetrics</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.ExportType</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.ProfilerAction</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.ProfilerActivity</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.ProfilerLevel</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler._ExperimentalConfig</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler._KinetoProfile</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.dynamic_profile.init</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.dynamic_profile.start</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.dynamic_profile.step</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.profile.disable_profiler_in_child_thread</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.profile.enable_profiler_in_child_thread</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.profile</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.profiler.analyse</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.schedule</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.supported_activities</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.supported_ai_core_metrics</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.supported_export_type</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.supported_profiler_level</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.reset_stream_limit</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.scatter_update</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.scatter_update_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.set_device_limit</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.set_stream_limit</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>    
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.get_cann_version</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.reset_thread_affinity</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.set_thread_affinity</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch.distributed.ProcessGroupHCCL</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch.distributed.is_hccl_available</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu._npu_dropout</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.distributed.all_gather_into_tensor_uneven</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.distributed.reinit_process_group</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.jit.optimize</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.aclnn.version</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.check_uce_in_memory</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.clear_npu_overflow_flag</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.config.allow_internal_format</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.get_amp_supported_dtype</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.get_autocast_dtype</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.get_mm_bmm_format_nd</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.is_autocast_enabled</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.is_jit_compile_false</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.restart_device</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_aoe</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_autocast_dtype</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_autocast_enabled</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_mm_bmm_format_nd</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.stop_device</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.stress_detect</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.utils.is_support_inf_nan</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.utils.npu_check_overflow</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_deformable_conv2d</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_linear</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rms_norm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_swiglu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.FlopsCounter</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.get_part_combined_tensor</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.is_combined_tensor_valid</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.npu_combine_tensors</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.save_async</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fusion_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_cross_entropy_loss</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_swiglu_quant_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.2.0版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rms_norm_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_format_cast</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_format_cast_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_lightning_indexer</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_compile_mode</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.enable_deterministic_with_backward</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.disable_deterministic_with_backward</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.Event().recorded_time()</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.ExternalEvent().record()</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.ExternalEvent().reset()</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.ExternalEvent().wait()</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_matmul_all_to_all</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_attention_update</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_all_to_all_matmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_scaled_masked_softmax</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.9.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.10.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.11.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.12.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.13.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.14.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
  </tbody>
</table>

> [!NOTE]
>
> TorchNPU新增部分API支持及特性支持，具体可参考《[自定义API](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/zh/custom_APIs/overview.md)》或《[原生API](../zh/api/native_api/pytorch_2-14-0/overview.md)》。

### 已解决问题

<table><tbody><tr id="row098217197105"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.1.1"><p id="p109824198109">问题描述</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.1.1 "><p id="p9982131912103"><strong id="b59839199105">现象</strong>：开启multi_stream_lazy_reclaim特性时造成随机的Core Dump。</p>
<p id="p15983141916104"><strong id="b1598312196108">影响</strong>：multi_stream_lazy_reclaim特性不可用。</p>
</td>
</tr>
<tr id="row1298311191102"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.2.1"><p id="p109831119201013">严重级别</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.2.1 "><p id="p18983019161017">严重</p>
</td>
</tr>
<tr id="row598371901017"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.3.1"><p id="p19833192101">根因分析</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.3.1 "><p id="p1798319199103">在特定场景下，process_events()中free_block()调用try_merge_blocks()可能提前释放已被get_free_block()取出但尚未标记为allocated的params.block。导致后续代码访问已释放内存（Use-After-Free），引发Core Dump。</p>
</td>
</tr>
<tr id="row1298318191109"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.4.1"><p id="p1798321961013">解决方案</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.4.1 "><p id="p119831219181019">1. 在get_free_block返回Block后，立即将其状态标记为allocated。<br>2. 将process_events中判断sum > kLazyQuerySize的逻辑提前至get_free_block调用之前执行。</p>
</td>
</tr>
<tr id="row1198341919103"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.5.1"><p id="p9983219181017">修改影响</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.5.1 "><p id="p15983119101017">修复后，该特性正常使用，Core Dump问题被修复。</p>
</td>
</tr>
</tbody>
</table>

### 遗留问题

无

## 升级影响

### 升级过程中对现行系统的影响

- 对业务的影响

    软件版本升级过程中会导致业务中断。

- 对网络通信的影响

    对通信无影响。

### 升级后对现行系统的影响

无

## 版本配套文档

|文档名称|内容简介|更新说明|
|---|---|---|
|《[软件安装](../zh/installation_guide/references/building_from_source.md)》|提供在昇腾设备安装PyTorch框架训练环境，以及升级、卸载等操作。|&#8226; 新增适配PyTorch 2.13.0和PyTorch 2.14.0。<br>&#8226; 新增编译加速和使用Clang编译文档。|
|《[TorchNPU概述](../zh/user_guide/product_overview.md)》|TorchNPU插件是基于昇腾的深度学习适配框架，使昇腾NPU可以支持PyTorch框架，为PyTorch框架的使用者提供昇腾AI处理器的超强算力。|无更新 |
|《[快速入门](../zh/user_guide/quick_start.md)》|提供了一个简单的模型迁移样例，采用了最简单的自动迁移方法，帮助用户快速体验GPU模型脚本迁移到昇腾NPU上的流程。|优化快速入门文档。|
|《[Torch.compile](../zh/user_guide/torch_compile/overview.md)》|通过“动态图捕获+静态图优化+高效代码生成”的方式显著加速模型训练和推理任务。|&#8226; Inductor后端新增Ascend C编译器。<br>&#8226; 新增PyTorch官方参考文档链接。  |
|《[配套软件库](./user_guide/libraries.md)》|为TorchNPU提供扩展能力的配套软件库。|新增Triton Asend组件。|
|《[故障处理](../zh/user_guide/troubleshooting/troubleshooting_process.md)》|以开发者在执行推理、训练过程中可能遇到的各类异常故障现象为入口，提供自助式问题定位、问题处理方法，方便开发者快速定位并解决故障。|无更新|
|《[原生API](../zh/api/native_api/pytorch_2-12-0/overview.md)》|提供PyTorch 2.12.0/2.11.0/2.10.0/2.9.0/2.7.1版本原生API在昇腾设备上的支持情况。|&#8226; 新增PyTorch 2.13.0和PyTorch 2.14.0原生API支持清单。<br>&#8226; 内容格式对齐PyTorch社区。 |
|《[自定义API](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/overview.md)》|提供TorchNPU自定义API的函数原型、功能说明、参数说明与调用示例等。|&#8226; 新增适配PyTorch 2.13.0和PyTorch 2.14.0。<br>&#8226; 具体接口变更请参考[接口变更说明](#接口变更说明)。|
|《[环境变量](../zh/api/environment_variable/env_variable_list.md)》|在TorchNPU训练和在线推理过程中可使用的环境变量。|&#8226; 新增PyTorch环境变量对照表。<br>&#8226; 调整部分目录结构。|
|《[内存管理](../zh/developer_notes/memory_management/memory_resource_overview.md)》|TorchNPU在内存管理方面构建了一套完整的体系，既深度集成了PyTorch原生的内存管理机制，又针对昇腾NPU硬件特性提供了多项独有的优化能力。|内容独立且优化。|
|《[分布式](../zh/developer_notes/distributed/distributed_overview.md)》|支持的核心特性，涵盖并行策略、分片原语、通信策略、分布式启动与容错等关键能力，并说明各项特性在NPU上的使用方式以及和原生PyTorch的异同。|&#8226; 分布式章节独立展示。<br>&#8226; 新增概述章节。<br>&#8226; 新增并行策略特性。<br>&#8226; 新增通信策略特性。<br>&#8226; 重构通信域参数。 |
|《[算子下发](../zh/developer_notes/operator_dispatch/automatic_core_binding.md)》|通过自动绑核、Stream级TaskQueue并行下发和编译优化，全面提升TorchNPU下发及程序运行性能。|&#8226; 分布式章节独立展示。<br>&#8226; 新增taskqueue特性。|
|《[NPUGraph](../zh/developer_notes/npugraph.md)》|NPUGraph是一种在Eager Mode（单算子执行模式）下使用的静态图捕获技术，将一系列NPU内核定义并封装为一个单元（即操作图），通过单一CPU操作启动多个NPU操作，从而减少启动开销。|内容独立且优化。|
|《[故障诊断](../zh/developer_notes/fault_diagnosis/feature_value_detection.md)》|在不影响大模型训练性能和精度的前提下，通过基于通信流的静默数据错误特征值检测技术，实现精度问题的快速稳定识别。|内容独立且优化。|
|《[自定义算子适配开发](../zh/developer_notes/custom_operator_adaptation/opplugin_operator_adaptation/adaptation_overview_opplugin.md)》|基于OpPlugin插件或C++ extensions的方式编写并调用自定义算子。|内容独立且优化。|
|《[TorchAir](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/overview.md)》|作为昇腾TorchNPU的图模式能力扩展库，提供昇腾设备亲和的torch.compile图模式后端，实现PyTorch网络在昇腾NPU上的图模式推理加速和优化。|&#8226; npugraph_ex功能增强：新增支持SuperKernel融合优化功能、force_recapture功能、图捕获安全策略配置等。<br>&#8226; GE图模式功能增强：扩展npu_stream_switch接口，支持指定并发策略等。<br>&#8226; 新增支持<term>Ascend 950DT</term>相关内容。|
|《[安全声明](../zh/reference/security_statement.md)》|提供了TorchNPU、OpPlugin、TorchAir和Ascend Extension for TensorPipe组件的软件版本、系统加固要求、安全配置（数据存储、调试接口、运行环境等）、权限配置、防火墙等设置。|例行更新。|

## 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-09-19 08:00:00.0|2026-09-20|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-09-20 10:13:00.0|2026-09-20|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.101048|2026-09-20|无病毒，无恶意|

## 漏洞修补列表

无

## 修订记录

|文档|发布日期|修改说明|
|--|--|--|
|01|2026-10-15|第一次正式发布|
