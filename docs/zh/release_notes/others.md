# 其他

## 已解决问题

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



## 遗留问题

<table><tbody><tr id="row098217197105"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.1.1"><p id="p109824198109">问题描述</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.1.1 "><p id="p9982131912103"><strong id="b59839199105">现象</strong>：部分API的某些dtype不支持（API具体支持的dtype信息可参考<span id="ph1521732894415"><a href="https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/overview.md">《Ascend Extension for PyTorch 自定义API参考》</a></span>或<span id="ph2608172172913"><a href="https://gitcode.com/Ascend/pytorch/blob/v2.9.0-7.3.0/docs/zh/native_apis/overview.md">《PyTorch 原生API支持度》</a></span>）</p>
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


