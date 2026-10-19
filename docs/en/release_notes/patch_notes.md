# 26.1.1 Patch Notes

## Basic Information

<table><tbody><tr id="row135479428341"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.1.1"><p id="p125478428345">Patch Number</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.1.1 "><p id="p3547142103415"><span id="ph4778145519911">26.1.1</span></p>
</td>
</tr>
<tr id="row11547114203412"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.2.1"><p id="p17547142103418">Product Base Version</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.2.1 "><p id="p2547184216342"><span id="ph1414342615376">26.1.0</span></p>
</td>
</tr>
<tr id="row854711422349"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.3.1"><p id="p354754216341">Release Date</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.3.1 "><p id="p2547114214349">September 1, 2026</p>
</td>
</tr>
</tbody>
</table>

## Download Address

Software package download address: [Quick Installation](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=182&ids=89dda9ba9de741349efa03687a487678%2C202%2C106%2C1%2C6%2C177%2C)

## Version Mapping

The version mapping table for firmware and drivers is related to all Ascend hardware and CANN versions. For specific selection, refer to the [CANN Release Notes](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/releasenote/9.1.0/release-notes.md).

To extend TorchNPU capabilities, Ascend provides self-developed plugins. For their version requirements, refer to [Supported Software Libraries](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.1.0/zh/supported_suites_and_third_party_libraries/supported_suites_and_third_party_libraries.md).

The version mapping between related products is shown in the following table:

|TorchNPU Code Branch Name|PyTorch Version|TorchNPU Version|TorchNPU Installation Package Version|CANN Version|Python Version|
|--|--|--|--|--|--|
|v2.7.1-26.1.0|2.7.1|26.1.1|2.7.1.post10|9.1.X|Python3.9.*x*, Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|
|v2.9.0-26.1.0|2.9.0|26.1.1|2.9.0.post8|9.1.X|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|
|v2.10.0-26.1.0|2.10.0|26.1.1|2.10.0.post6|9.1.X|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|
|v2.11.0-26.1.0|2.11.0|26.1.1|2.11.0.post2|9.1.X|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|
|v2.12.0-26.1.0|2.12.0|26.1.1|2.12.0.post2|9.1.X|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|

## Version Compatibility

> [!NOTE]
>
> In the table, "Y" means compatible.

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
    <th class="tg-amwm" colspan="3">CANN Version</th>
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

## Patch Installation Impact

### Impact on the Current System During Installation

None

### Impact on the Current System After Installation

None

## Fixed Issues

- Fixed the core dump issue that occurred when profiler collection was enabled on PyTorch 2.12 and later versions.
- Fixed the installation and compilation error that occurred when TorchNPU 26.1.0 was used with CANN 9.0.0 and earlier versions.
- The Triton Ascend version bundled with TorchNPU 26.1.0 has been updated to 3.2.2.

## Known Issues

None

## Virus Scan Results and Vulnerability Patch List

### Virus Scan Results

|Antivirus Software Name|Antivirus Software Version|Virus Database Version|Scan Time|Scan Result|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-08-24 08:00:00.0|2026-08-25|Virus-free and malware-free|
|Kaspersky|12.0.0.6672|2026-08-25 10:10:00.0|2026-08-25|Virus-free and malware-free|
|Bitdefender|7.5.1.200224|7.101277|2026-08-25|Virus-free and malware-free|

### Vulnerability Patch List

No vulnerability fixes in this release.
