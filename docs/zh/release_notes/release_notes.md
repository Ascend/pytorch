# 版本说明

## 版本配套说明

### 产品版本信息

<table><tbody><tr id="row135479428341"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.1.1"><p id="p125478428345">产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.1.1 "><p id="p3547142103415"><span id="ph4778145519911">Ascend Extension for PyTorch</span></p>
</td>
</tr>
<tr id="row11547114203412"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.2.1"><p id="p17547142103418">产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.2.1 "><p id="p2547184216342"><span id="ph1414342615376">26.0.0</span></p>
</td>
</tr>
<tr id="row854711422349"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.3.1"><p id="p354754216341">版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.3.1 "><p id="p2547114214349">正式版本</p>
</td>
</tr>
<tr id="row754461214611"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.4.1"><p id="p155445122062">发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.4.1 "><p id="p135443128613">2026年4月</p>
</td>
</tr>
<tr id="row954744243418"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.5.1"><p id="p15471742193419">维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.5.1 "><p id="p1154734212344">参考<a href="https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/README.zh.md#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5">分支维护策略</a></p>
</td>
</tr>
</tbody>
</table>

### 相关产品版本配套说明

Ascend Extension for PyTorch代码分支名称采用 **\{PyTorch版本\}-\{昇腾版本\}** 的命名规则，前者为Ascend Extension for PyTorch匹配的PyTorch版本，后者为Ascend Extension for PyTorch版本，详细匹配如下表：

|CANN版本|PyTorch版本|Ascend Extension for PyTorch版本|Ascend Extension for PyTorch代码分支名称|Ascend Extension for PyTorch安装包版本|Python版本|
|--|--|--|--|--|--|
|商用版：9.0.0<br>社区版：9.0.0|2.7.1|26.0.0|v2.7.1-26.0.0|2.7.1.post4|Python3.9.*x*、Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|商用版：9.0.0<br>社区版：9.0.0|2.8.0|26.0.0|v2.8.0-26.0.0|2.8.0.post4|Python3.9.*x*、Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|商用版：9.0.0<br>社区版：9.0.0|2.9.0|26.0.0|v2.9.0-26.0.0|2.9.0.post2|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|商用版：9.0.0<br>社区版：9.0.0|2.10.0|26.0.0|v2.10.0-26.0.0|2.10.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|

## 版本兼容性说明

|Ascend Extension for PyTorch版本|CANN版本|网上相关产品版本兼容性|
|--|--|--|
|26.0.0|CANN 9.0.0<br>CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1|-|
|7.3.0|CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0|Driving SDK 7.3.0兼容CANN 8.1.RC1、CANN 8.2.RC1、CANN 8.3.RC1、CANN 8.5.0|
|7.2.0|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|Driving SDK 7.2.RC1兼容CANN 8.1.RC1、CANN 8.2.RC1、CANN 8.3.RC1|
|7.1.0|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|Driving SDK 7.1.RC1兼容CANN 8.0.0、CANN 8.1.RC1和CANN 8.2.RC1|

## 更新说明

### 新增特性

<table>
  <thead align="left">
    <tr>
      <th class="cellrowborder" valign="top" width="18.801880188018803%" id="mcps1.1.4.1.1">组件</th>
      <th class="cellrowborder" valign="top" width="32.603260326032604%" id="mcps1.1.4.1.2">描述</th>
      <th class="cellrowborder" valign="top" width="48.5948594859486%" id="mcps1.1.4.1.3">目的</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="cellrowborder" rowspan="6" valign="top" width="18.801880188018803%" headers="mcps1.1.4.1.1"><span>Ascend Extension for PyTorch</span>（即torch-npu）</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">P2P通信支持group下发。</td>
      <td class="cellrowborder" valign="top" width="48.5948594859486%" headers="mcps1.1.4.1.3">batch_isend_irecv支持group下发方式。</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">DTensor策略扩展。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">新增多类算子切分支持，算子覆盖率提升至99%以上。</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">跨流内存延迟释放优化。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">在内存充足时，减少跨流内存释放状态查询开销，缩短运行时间，提升执行性能。</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">D2D拷贝的Event使用Pool管理。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">对跨设备拷贝中的Event进行池化管理，减少频繁创建开销，提升性能。</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Event能力增强，支持跨进程与跨设备使用。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">在跨进程共享内存和跨设备拷贝场景中支持使用Event同步，提升整体性能。</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">host allocator支持背景线程优化query时间。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">使用背景线程定时查询Event的完成情况，提升整体内存分配性能。</td>
    </tr>
  </tbody>
</table>

### 删除特性

无

### 接口变更说明

本章节的接口变更说明包括新增、修改、废弃和删除。接口变更只体现代码层面的修改，不包含文档本身在语言、格式、链接等方面的优化改进。

- 新增：表示此次版本新增的接口。
- 修改：表示本接口相比于上个版本有修改。
- 废弃：表示该接口自作出废弃声明的版本起停止演进，且在声明一年后可能被移除。
- 删除：表示该接口在此次版本被移除。

**表 1** Ascend Extension for PyTorch接口变更汇总

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
      <td class="cellrowborder" rowspan="18" valign="top" width="11.53%" headers="mcps1.2.6.1.1">v2.7.1</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.NpuGraphOpHandler</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">&#8226; torch_npu功能重构<br>&#8226; 不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_block_sparse_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dense_lightning_indexer_grad_kl_loss</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dense_lightning_indexer_softmax_lse</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_floyd_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fusion_attention_v3</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_matmul_gelu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.save_npugraph_tensor</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 8.5.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_attention_to_ffn</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_attention_update</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_recurrent_gated_delta_rule</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.matmul.cube_math_type</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 8.5.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_swiglu_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rms_norm_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 8.3.0RC1及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_clipped_swiglu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fusion_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">&#8226; 新增dropout_mask、seed、offset可选参数<br>&#8226; 不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.8.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.9.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.10.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
  </tbody>
</table>

> [!NOTE]  
> Ascend Extension for PyTorch新增部分API支持及特性支持，具体可参考《[Ascend Extension for PyTorch 自定义API参考](https://gitcode.com/Ascend/op-plugin/tree/26.0.0/docs/context/overview.md)》或《[PyTorch 原生API支持度](../native_apis/pytorch_2-10-0/overview.md)》。

### 已解决问题

<table><tbody><tr id="row098217197105"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.1.1"><p id="p109824198109">问题描述</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.1.1 "><p id="p9982131912103"><strong id="b59839199105">现象</strong>：在特定场景下，可能出现NN process问题，导致两张IC（计算实例）网络发生OOM。</p>
<p id="p15983141916104"><strong id="b1598312196108">影响</strong>：部分环境中可能出现NN process报错。</p>
</td>
</tr>
<tr id="row1298311191102"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.2.1"><p id="p109831119201013">严重级别</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.2.1 "><p id="p18983019161017">一般</p>
</td>
</tr>
<tr id="row598371901017"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.3.1"><p id="p19833192101">根因分析</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.3.1 "><p id="p1798319199103">get_device_properties接口通过cudaGetDeviceProperties接口调用获取，NPU上调用多个接口获取对应属性。其中设备内存信息依赖aclGetMemInfo获取，但该接口必须绑定context才能调用，导致持续占用0卡内存。</p>
</td>
</tr>
<tr id="row1298318191109"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.4.1"><p id="p1798321961013">解决方案</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.4.1 "><p id="p119831219181019">get_device_properties接口对齐aclrtGetMenInfo接口。</p>
</td>
</tr>
<tr id="row1198341919103"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.5.1"><p id="p9983219181017">修改影响</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.5.1 "><p id="p15983119101017">修复后，get_device_properties接口对齐了aclrtGetMenInfo接口，消除了潜在的资源占用风险。</p>
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
|《[Ascend Extension for PyTorch 软件安装指南](../installation_guide/installation_description.md)》|提供在昇腾设备安装PyTorch框架训练环境，以及升级、卸载等操作。|&#8226; 新增适配PyTorch 2.10.0。<br>&#8226; 新增支持Python 3.13。|
|《[PyTorch 训练模型迁移调优指南](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/overview.md)》|包含模型的迁移及调优、精度问题定位、性能问题解决等指导，并提供了常用模型案例库。|新增适配PyTorch 2.10.0。|
|《[PyTorch 框架特性指南](../framework_feature_guide_pytorch/overview.md)》|基于Ascend Extension for PyTorch提供昇腾AI处理器的超强算力，从内存优化、报错定位、高性能计算等方面打造一系列独有特性。|&#8226; 新增“NPUGraph”相关内容。<br>&#8226; 更新“基于C++ extension算子适配开发”相关内容。<br>&#8226; 优化“基于OpPlugin算子适配开发”相关内容。|
|《[PyTorch 图模式使用指南(TorchAir)](https://gitcode.com/Ascend/torchair/blob/26.0.0/docs/zh/overview.md)》|作为昇腾Ascend Extension for PyTorch的图模式能力扩展库，提供昇腾设备亲和的torch.compile图模式后端，实现PyTorch网络在昇腾NPU上的图模式推理加速和优化。|&#8226; 新增npugraph_ex后端的图模式，通过Capture&Replay方式实现任务一次捕获多次执行。原reduce-overhead模式通过config.mode配置图编辑后端的方式将不再演进，也不再推荐使用。<br>&#8226; 增强GE模式功能，包括指定dump算子范围、图内标定SuperKernel范围等。|
|《[Ascend Extension for PyTorch 自定义API参考](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/zh/custom_APIs/overview.md)》|提供Ascend Extension for PyTorch自定义API的函数原型、功能说明、参数说明与调用示例等。|&#8226; 新增适配PyTorch 2.10.0。<br>&#8226; 具体接口变更请参考[接口变更说明](#接口变更说明)。|
|《[PyTorch 原生API支持度](../native_apis/pytorch_2-10-0/overview.md)》|提供PyTorch 2.10.0/2.9.0/2.8.0/2.7.1版本原生API在昇腾设备上的支持情况。|新增PyTorch 2.10.0原生API支持清单。|
|《[套件与三方库支持清单](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/zh/supported_suites_and_third_party_libraries/supported_suites_and_third_party_libraries.md)》|介绍昇腾设备支持的模型套件与加速库、昇腾已原生支持的第三方库和昇腾自研插件。|无新增。|
|《[环境变量参考](../environment_variable_reference/env_variable_list.md)》|在Ascend Extension for PyTorch训练和在线推理过程中可使用的环境变量。|&#8226; 新增“TORCH_NPU_USE_COMPATIBLE_IMPL”。<br>&#8226; 新增“TORCH_NPU_LOGS”。<br>&#8226; 新增“TORCH_NPU_LOGS_FILTER”。<br>&#8226; 新增“TORCH_NPU_DEVICE_CAPABILITY”。<br>&#8226; 新增“TORCH_TRANSFER_TO_NPU”。<br>&#8226; 新增“TORCHINDUCTOR_NPU_BACKEND”。<br>&#8226; 新增“INDUCTOR_ASCEND_CHECK_ACCURACY”。|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-04-01 08:00:00.0|2026-04-02|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-04-02 10:05:00.0|2026-04-02|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.100588|2026-04-02|无病毒，无恶意|

### 漏洞修补列表

本版本无漏洞修复。
