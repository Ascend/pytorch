# 26.1.1补丁说明

## 补丁基本信息

<table><tbody><tr id="row135479428341"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.1.1"><p id="p125478428345">补丁号</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.1.1 "><p id="p3547142103415"><span id="ph4778145519911">26.1.1</span></p>
</td>
</tr>
<tr id="row11547114203412"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.2.1"><p id="p17547142103418">产品基础版本</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.2.1 "><p id="p2547184216342"><span id="ph1414342615376">26.1.0</span></p>
</td>
</tr>
<tr id="row854711422349"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.3.1"><p id="p354754216341">发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.3.1 "><p id="p2547114214349">2026年9月1日</p>
</td>
</tr>
</tbody>
</table>

## 版本下载地址

软件包下载地址：[快速安装](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=175&ids=89dda9ba9de741349efa03687a487678%2C202%2C106%2C1%2C6%2C177%2C)

## 相关产品版本配套说明

固件和驱动的版本配套表与所有的昇腾硬件及CANN版本相关，具体选择请参考[CANN版本说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/releasenote/9.1.0/release-notes.md)。

为扩展TorchNPU能力，昇腾提供的自研插件，其版本要求说明请参考[配套软件库](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.1.0/zh/supported_suites_and_third_party_libraries/supported_suites_and_third_party_libraries.md)。

相关产品版本配套关系见下表：

|TorchNPU代码分支名称|PyTorch版本|TorchNPU版本|TorchNPU安装包版本|CANN版本|Python版本|
|--|--|--|--|--|--|
|v2.7.1-26.1.0|2.7.1|26.1.1|2.7.1.post10|9.1.X|Python3.9.*x*、Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|v2.9.0-26.1.0|2.9.0|26.1.1|2.9.0.post8|9.1.X|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|v2.10.0-26.1.0|2.10.0|26.1.1|2.10.0.post6|9.1.X|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|v2.11.0-26.1.0|2.11.0|26.1.1|2.11.0.post2|9.1.X|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|v2.12.0-26.1.0|2.12.0|26.1.1|2.12.0.post2|9.1.X|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|

## 版本兼容性信息

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
    <th class="tg-amwm" colspan="3">CANN版本</th>
  </tr>
  <tr>
    <th class="tg-c3ow">8.5.X</th>
    <th class="tg-c3ow">9.0.X</th>
    <th class="tg-c3ow">9.1.X</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">7.3.X</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
  </tr>
  <tr>
    <td class="tg-baqh">26.0.X</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
  </tr>
    <tr>
    <td class="tg-baqh">26.1.X</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
  </tr>
</tbody>
</table>

## 安装补丁的影响

### 安装过程中对现行系统的影响

无

### 安装后对现行系统的影响

无

## 已修复问题

- 解决了PyTorch 2.12及以上版本，开启profiler采集后出现core dump的问题。
- 解决了TorchNPU 26.1.0版本搭配CANN 9.0.0及之前版本存在的安装编译错误问题。
- TorchNPU 26.1.0版本配套Triton Ascend版本更新为3.2.2。

## 遗留问题

无

## 病毒扫描结果及漏洞修补列表

### 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-08-24 08:00:00.0|2026-08-25|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-08-25 10:10:00.0|2026-08-25|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.101277|2026-08-25|无病毒，无恶意|

### 漏洞修补列表

本版本无漏洞修复。
