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
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.2.1 "><p id="p2547184216342"><span id="ph1414342615376">7.3.0</span></p>
</td>
</tr>
<tr id="row854711422349"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.3.1"><p id="p354754216341">版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.3.1 "><p id="p2547114214349">正式版本</p>
</td>
</tr>
<tr id="row754461214611"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.4.1"><p id="p155445122062">发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.4.1 "><p id="p135443128613">2026年1月</p>
</td>
</tr>
<tr id="row954744243418"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.5.1"><p id="p15471742193419">维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.5.1 "><p id="p1154734212344">1年</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]  
> 有关Ascend Extension for PyTorch的版本维护，具体请参见[分支维护策略](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-7.3.0/README.zh.md#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5)。

### 相关产品版本配套说明

Ascend Extension for PyTorch代码分支名称采用 **\{PyTorch版本\}-\{昇腾版本\}** 的命名规则，前者为Ascend Extension for PyTorch匹配的PyTorch版本，后者为Ascend Extension for PyTorch版本，详细匹配如下表：

**表 1**  PyTorch软件版本配套关系

|CANN版本|PyTorch版本|Ascend Extension for PyTorch版本|Ascend Extension for PyTorch代码分支名称|Ascend Extension for PyTorch安装包版本|Python版本|
|--|--|--|--|--|--|
|商用版：8.5.0<br>社区版：8.5.0|2.6.0|7.3.0|v2.6.0-7.3.0|2.6.0.post5|Python3.9.*x*、Python3.10.*x*、Python3.11.*x*、Python3.12.*x*|
|商用版：8.5.0<br>社区版：8.5.0|2.7.1|7.3.0|v2.7.1-7.3.0|2.7.1.post2|Python3.9.*x*、Python3.10.*x*、Python3.11.*x*、Python3.12.*x*|
|商用版：8.5.0<br>社区版：8.5.0|2.8.0|7.3.0|v2.8.0-7.3.0|2.8.0.post2|Python3.9.*x*、Python3.10.*x*、Python3.11.*x*、Python3.12.*x*|
|商用版：8.5.0<br>社区版：8.5.0|2.9.0|7.3.0|v2.9.0-7.3.0|2.9.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*|


**表 2**  组件间配套表

|Ascend Extension for PyTorch版本|Ascend Extension for PyTorch代码分支名称|Ascend Extension for PyTorch安装包版本|DrivingSDK|
|--|--|--|--|
|7.3.0|v2.6.0-7.3.0|2.6.0.post5|7.3.0|
|7.3.0|v2.7.1-7.3.0|2.7.1.post2|7.3.0|
|7.3.0|v2.8.0-7.3.0|2.8.0.post2|7.3.0|
|7.3.0|v2.9.0-7.3.0|2.9.0|7.3.0|

## 版本兼容性说明

|Ascend Extension for PyTorch版本|CANN版本|网上相关产品版本兼容性|
|--|--|--|
|7.3.0|CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0|DrivingSDK 7.3.0兼容CANN 8.1.RC1、CANN 8.2.RC1、CANN 8.3.RC1、CANN 8.5.0|
|7.2.0|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|DrivingSDK 7.2.RC1兼容CANN 8.1.RC1、CANN 8.2.RC1、CANN 8.3.RC1|
|7.1.0|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|DrivingSDK 7.1.RC1兼容CANN 8.0.0、CANN 8.1.RC1和CANN 8.2.RC1|
|7.0.0|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|DrivingSDK 7.0.RC1兼容CANN 8.0.0和CANN 8.1.RC1|

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
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">集合通信内存复用优化</td>
      <td class="cellrowborder" valign="top" width="48.5948594859486%" headers="mcps1.1.4.1.3">新增erase_record_stream的增强模式，内存复用率更高</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">host allocator对齐社区</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">复用社区已有的host allocator机制，增强host allocator能力</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">新增支持PyTorch 2.9.0</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">通用能力，与社区同步发布</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">新增支持Python 3.12</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">通用能力，支持3.12版本的Python</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">新增支持symmetric memory接入shmem</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">对齐nvshmem，适配接入NPU的shmem能力</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">通信域异常检测能力增强</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">HCCL异常检测与watchdog解耦，支持未下发通信算子时也对hccl链路状态进行检测</td>
    </tr>
    <tr>
      <td class="cellrowborder" rowspan="4" valign="top" width="18.801880188018803%" headers="mcps1.1.4.1.1">Driving SDK</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">新增Pi0.5模型适配</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.3">适配业界主流VLA模型，支持具身智能和自动驾驶场景</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">新增GR00T-N1.5模型适配</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">适配业界主流VLA模型，支持具身智能和自动驾驶场景</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">新增VGGT模型适配</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">适配业界主流世界模型，支持具身智能和自动驾驶场景</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">新增自驾典型模型环境配置脚本</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">提升Driving SDK易用性</td>
    </tr>
  </tbody>
</table>

### 删除特性

无

### 接口变更说明

本章节的接口变更说明包括新增、修改、废弃和删除。接口变更只体现代码层面的修改，不包含文档本身在语言、格式、链接等方面的优化改进。

-   新增：表示此次版本新增的接口。
-   修改：表示本接口相比于上个版本有修改。
-   废弃：表示该接口自作出废弃声明的版本起停止演进，且在声明一年后可能被移除。
-   删除：表示该接口在此次版本被移除。

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
      <td class="cellrowborder" rowspan="11" valign="top" width="11.53%" headers="mcps1.2.6.1.1">v2.7.1</td>
      <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.2">torch_npu.npu_grouped_matmul_swiglu_quant_v2</td>
      <td class="cellrowborder" valign="top" width="15.22%" headers="mcps1.2.6.1.3">自定义接口</td>
      <td class="cellrowborder" valign="top" width="15.32%" headers="mcps1.2.6.1.4">新增</td>
      <td class="cellrowborder" valign="top" width="20.25%" headers="mcps1.2.6.1.5">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_gelu_mul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_sim_exponential_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_lightning_indexer</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_sparse_flash_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_sparse_lightning_indexer_grad_kl_loss</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fusion_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">参数变更，新增可选入参sink</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mla_prolog_v3</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_lightning_indexer</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_kv_quant_sparse_flash_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_recurrent_gated_delta_rule_functional</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.6.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.8.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5"></a>变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.9.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
  </tbody>
</table>

> [!NOTE]  
> Ascend Extension for PyTorch新增部分API支持及特性支持，具体可参考《[Ascend Extension for PyTorch 自定义API参考](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/overview.md)》或《[PyTorch 原生API支持度](../native_apis/pytorch_2-9-0/overview.md)》。


### 已解决问题

<table><tbody><tr id="row098217197105"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.1.1"><p id="p109824198109">问题描述</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.1.1 "><p id="p9982131912103"><strong id="b59839199105">现象</strong>：部分场景出现std::bad_alloc或者invalid pointer，查看coredump堆栈发现为std::regex后引入</p>
<p id="p15983141916104"><strong id="b1598312196108">影响</strong>：部分环境中调用set_device接口可能报错</p>
</td>
</tr>
<tr id="row1298311191102"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.2.1"><p id="p109831119201013">严重级别</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.2.1 "><p id="p18983019161017">一般</p>
</td>
</tr>
<tr id="row598371901017"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.3.1"><p id="p19833192101">根因分析</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.3.1 "><p id="p1798319199103">torch_npu使用的ABI版本与PyTorch保持一致，在部分场景可能因为系统内其他ABI不一致的so影响，出现跨ABI版本的so间的调用，导致未知错误</p>
</td>
</tr>
<tr id="row1298318191109"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.4.1"><p id="p1798321961013">解决方案</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.4.1 "><p id="p119831219181019"><span id="ph169431461939">通过添加编译选项-Bsymbolic-functions控制优先查找库内符号，避免so间跨ABI调用导致的未知错误</p>
</td>
</tr>
<tr id="row1198341919103"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.5.1"><p id="p9983219181017">修改影响</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.5.1 "><p id="p15983119101017">共享库符号绑定规则将从默认外部优先更改为内部优先，不影响torch_npu内部功能，如果外部存在劫持torch_npu内部符号的场景可能会失效</p>
</td>
</tr>
</tbody>
</table>



### 遗留问题

<table><tbody><tr id="row098217197105"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.1.1"><p id="p109824198109">问题描述</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.1.1 "><p id="p9982131912103"><strong id="b59839199105">现象</strong>：部分API的某些dtype不支持（API具体支持的dtype信息可参考《<span id="ph1521732894415"><a href="https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/overview.md">Ascend Extension for PyTorch 自定义API参考</a></span>》或《<span id="ph2608172172913"><a href="../native_apis/pytorch_2-9-0/overview.md">PyTorch 原生API支持度</a></span>》）</p>
<p id="p15983141916104"><strong id="b1598312196108">影响</strong>：API使用不支持的dtype会报错</p>
</td>
</tr>
<tr id="row1298311191102"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.2.1"><p id="p109831119201013">严重级别</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.2.1 "><p id="p18983019161017">一般</p>
</td>
</tr>
<tr id="row598371901017"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.3.1"><p id="p19833192101">规避和应急措施</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.3.1 "><p id="p1798319199103">建议使用支持的其他dtype进行规避</p>
</td>
</tr>
<tr id="row1298318191109"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.4.1"><p id="p1798321961013">影响域</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.4.1 "><p id="p119831219181019"><span id="ph169431461939">Ascend Extension for PyTorch</span> 7.3.0</p>
</td>
</tr>
<tr id="row1198341919103"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.5.1"><p id="p9983219181017">解决进展</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.5.1 "><p id="p15983119101017">问题遗留至后续版本解决</p>
</td>
</tr>
</tbody>
</table>

## 升级影响

### 升级过程中对现行系统的影响

-   对业务的影响

    软件版本升级过程中会导致业务中断。

-   对网络通信的影响

    对通信无影响。

### 升级后对现行系统的影响

无

## 版本配套文档

|文档名称|内容简介|更新说明|
|---|---|---|
|《[Ascend Extension for PyTorch 软件安装指南](../installation_guide/installation_description.md)》|提供在昇腾设备安装PyTorch框架训练环境，以及升级、卸载等操作。|- 新增适配PyTorch 2.9.0。<br>- 新增支持Python 3.12。|
|《[PyTorch 训练模型迁移调优指南](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/PT_LMTMOG_0002.html)》|包含模型的迁移及调优、精度问题定位、性能问题解决等指导，并提供了常用模型案例库。|新增适配PyTorch 2.9.0。|
|《[PyTorch 框架特性指南](../framework_feature_guide_pytorch/overview.md)》|基于Ascend Extension for PyTorch提供昇腾AI处理器的超强算力，从内存优化、报错定位、高性能计算等方面打造一系列独有特性。|- 新增“Stream级TaskQueue并行下发”特性。<br>- 新增“PyTorch编译模式（torch.compile）”特性。|
|《[PyTorch 图模式使用指南(TorchAir)](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00004.html)》|作为昇腾Ascend Extension for PyTorch的图模式能力扩展库，提供昇腾设备亲和的torch.compile图模式后端，实现PyTorch网络在昇腾NPU上的图模式推理加速和优化。|- 增强基础功能，包括完整Debug信息Dump、自定义FX Pass等。<br>- 增强aclgraph功能，包括支持Stream级控、内存复用、FX pass配置等。<br>- 增强GE功能，包括算子不超时配置等。|
|《[Ascend Extension for PyTorch 自定义API参考](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/overview.md)》|提供Ascend Extension for PyTorch自定义API的函数原型、功能说明、参数说明与调用示例等。|- 新增适配PyTorch 2.9.0。<br>- 具体接口变更请参考[接口变更说明](#接口变更说明)。|
|《[PyTorch 原生API支持度](../native_apis/pytorch_2-9-0/overview.md)》|提供PyTorch 2.9.0/2.8.0/2.7.1/2.6.0版本原生API在昇腾设备上的支持情况。|新增PyTorch 2.9.0原生API支持清单。|
|《[套件与三方库支持清单](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/modparts/thirdpart_0003.html)》|介绍昇腾设备支持的模型套件与加速库、昇腾已原生支持的第三方库和昇腾自研插件。|新增原生支持的第三方库ms-swift。|
|《[环境变量参考](../environment_variable_reference/env_variable_list.md)》|在Ascend Extension for PyTorch训练和在线推理过程中可使用的环境变量。|- 新增“PER_STREAM_QUEUE”。<br>- 新增“MULTI_STREAM_MEMORY_REUSE”。|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-01-03 08:00:00.0|2025-12-06|无病毒，无恶意|
|Kasperaky|12.0.0.6672|2026-01-05 10:03:00.0|2025-12-06|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.100087|2025-12-06|无病毒，无恶意|


### 漏洞修补列表

本版本无漏洞修复。



