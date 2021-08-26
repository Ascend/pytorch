# PyTorch Release Notes 2.0.2
-   [Before You Start](#before-you-start)
-   [New Features](#new-features)
-   [Modified Features](#modified-features)
-   [Resolved Issues](#resolved-issues)
-   [Known Issues](#known-issues)
-   [Compatibility](#compatibility)
<h2 id="before-you-start">Before You Start</h2>

This framework is modified based on the open-source PyTorch 1.5.0 primarily developed by Facebook, inherits native PyTorch features, and uses NPUs for dynamic image training. Models are adapted by operator granularity, code can be reused, and current networks can be ported and used on NPUs with only device types or data types modified.

<h2 id="new-features">New Features</h2>

**Table  1**  Features supported by  PyTorch

<a name="t76c34275cbb74753970f7c5a9eb594fa"></a>
<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="10.489999999999998%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>Level-1 Feature</p>
</th>
<th class="cellrowborder" valign="top" width="26.3%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>Level-2 Feature</p>
</th>
<th class="cellrowborder" valign="top" width="63.21%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row171322953713"><td class="cellrowborder" rowspan="7" valign="top" width="10.489999999999998%" headers="mcps1.2.4.1.1 "><p id="p10237517181"><a name="p10237517181"></a><a name="p10237517181"></a>Adapted training models</p>
</td>
<td class="cellrowborder" valign="top" width="26.3%" headers="mcps1.2.4.1.2 "><p id="p6134294377"><a name="p6134294377"></a><a name="p6134294377"></a>YOLOV4</p>
</td>
<td class="cellrowborder" valign="top" width="63.21%" headers="mcps1.2.4.1.3 "><p id="p314533811397"><a name="p314533811397"></a><a name="p314533811397"></a>-</p>
</td>
</tr>
<tr id="row15990182233714"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p399019224374"><a name="p399019224374"></a><a name="p399019224374"></a>YOLOV3</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1714553816396"><a name="p1714553816396"></a><a name="p1714553816396"></a>-</p>
</td>
</tr>
<tr id="row5301255373"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1530325123719"><a name="p1530325123719"></a><a name="p1530325123719"></a>DB</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p5146193819390"><a name="p5146193819390"></a><a name="p5146193819390"></a>-</p>
</td>
</tr>
<tr id="row274203413712"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p197423343379"><a name="p197423343379"></a><a name="p197423343379"></a>RFCN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p15828941153919"><a name="p15828941153919"></a><a name="p15828941153919"></a>-</p>
</td>
</tr>
<tr id="row78671918163714"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p188671318183713"><a name="p188671318183713"></a><a name="p188671318183713"></a>CRNN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10828241123917"><a name="p10828241123917"></a><a name="p10828241123917"></a>-</p>
</td>
</tr>
<tr id="row16912221171"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p6913132676"><a name="p6913132676"></a><a name="p6913132676"></a>Densenset161</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p209131221474"><a name="p209131221474"></a><a name="p209131221474"></a>-</p>
</td>
</tr>
<tr id="row1016073314719"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p3160153320717"><a name="p3160153320717"></a><a name="p3160153320717"></a>Densenset191</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1516012331173"><a name="p1516012331173"></a><a name="p1516012331173"></a>-</p>
</td>
</tr>
<tr id="row9627165910386"><td class="cellrowborder" rowspan="4" valign="top" width="10.489999999999998%" headers="mcps1.2.4.1.1 "><p id="p1561535993811"><a name="p1561535993811"></a><a name="p1561535993811"></a>PyTorch features adapted to NPUs</p>
</td>
<td class="cellrowborder" valign="top" width="26.3%" headers="mcps1.2.4.1.2 "><p id="p136151659163819"><a name="p136151659163819"></a><a name="p136151659163819"></a>Basic framework functions</p>
</td>
<td class="cellrowborder" valign="top" width="63.21%" headers="mcps1.2.4.1.3 "><p id="p1661535983813"><a name="p1661535983813"></a><a name="p1661535983813"></a>Added the function of adapted operator development. For details, see the operator list.</p>
</td>
</tr>
<tr id="row7627155917380"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1561525916383"><a name="p1561525916383"></a><a name="p1561525916383"></a>Model accuracy analyzer</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p19616115953816"><a name="p19616115953816"></a><a name="p19616115953816"></a>Added model accuracy analyzers and supported training accuracy demarcation.</p>
</td>
</tr>
<tr id="row46269593383"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p0616559163819"><a name="p0616559163819"></a><a name="p0616559163819"></a>Ascend 710 AI Processor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p186161459193813"><a name="p186161459193813"></a><a name="p186161459193813"></a>Supported the online inference for the Ascend 710 AI Processor.</p>
</td>
</tr>
<tr id="row76261059153817"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p11616175953812"><a name="p11616175953812"></a><a name="p11616175953812"></a>OS compatibility</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p8935115971714"><a name="p8935115971714"></a><a name="p8935115971714"></a>Supported Ubunta 18.04.5 and openEuler 20.03 LTS.</p>
</td>
</tr>
</tbody>
</table>

<h2 id="modified-features">Modified Features</h2>

N/A

<h2 id="resolved-issues">Resolved Issues</h2>

N/A

<h2 id="known-issues">Known Issues</h2>

<a name="table1969972073016"></a>
<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>Known Issue</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>Data type</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPU does not support the input or output of the inf/nan data of the float16 type.</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>Data format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>Dimensions cannot be reduced when the format larger than 4D is used.</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>Restrictions on collective communication</p>
<p id="p0465121912402"><a name="p0465121912402"></a><a name="p0465121912402"></a></p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>The graphs executed on different devices in a training job must be the same.</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>Allocation at only 1, 2, 4, or 8 processors is supported.</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>Only the int8, int32, float16, and float32 data types are supported.</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex function</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>In the current version, Apex is implemented mainly using Python, and the customized optimization CUDA kernel in Apex is not supported.</p>
</td>
</tr>
</tbody>
</table>

<h2 id="compatibility">Compatibility</h2>

Atlas 800 \(model 9010\): CentOS 7.6/Ubuntu 18.04/BC-Linux 7.6/Debian 9.9/Debian 10/openEuler 20.03 LTS

Atlas 800 \(model 9000\): CentOS 7.6/Euler 2.8/Kylin v10/BC-Linux 7.6/openEuler 20.03 LTS

